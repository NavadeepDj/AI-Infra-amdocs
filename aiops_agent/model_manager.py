"""
model_manager.py — Singleton ModelManager for the Explainable AIOps Agent
=========================================================================
Loads all serialized .joblib models and the parquet dataset ONCE at startup.
Provides sub-20ms inference methods for downstream tools.

Key Design:
- Load-once caching: models stay in memory for the agent's lifetime
- Schema validation: verifies feature columns match feature_order.json
- Stale data detection: warns if dataset is older than threshold
- Graceful degradation: missing models marked unavailable, not crashing
"""

import json
import time
from datetime import datetime, timezone
import logging
import numpy as np
import pandas as pd
try:
    pd.set_option("future.no_silent_downcasting", True)
except (pd.errors.OptionError, AttributeError):
    pass
from pathlib import Path

import joblib
import shap

from . import config

logger = logging.getLogger("aiops_agent.model_manager")


def safe_int(val, default=0) -> int:
    """Safely convert a value to int, handling NaN and None."""
    if val is None or pd.isna(val):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


class SchemaValidationError(Exception):
    """Raised when dataset columns don't match the expected feature schema."""
    pass


class ModelManager:
    """
    Singleton-style model and data manager.
    Call __init__ once at agent startup. All tools reference the same instance.
    """

    def __init__(self):
        logger.info("Initializing ModelManager — loading models and dataset...")
        t0 = time.time()

        # ── Load Metadata ────────────────────────────────────────────────
        self._feature_meta = self._load_json(config.FEATURE_ORDER_PATH)
        self._thresholds = self._load_json(config.THRESHOLDS_PATH)

        self._iforest_features = self._feature_meta.get("isolation_forest_features", [])
        self._xgb_features = self._feature_meta.get("xgboost_failure_features", [])
        self._imputation_rules = self._feature_meta.get("imputation_rules", {})

        # ── Load Dataset ─────────────────────────────────────────────────
        self._df = self._load_dataset()
        self._server_list = sorted(self._df["machine_name"].unique().tolist())
        self._data_freshness = self._compute_data_freshness()

        # ── Load Models (Graceful Degradation) ───────────────────────────
        self._iforest = self._load_model(config.IFOREST_PATH, "Isolation Forest")
        self._xgb_3slot = self._load_model(config.XGB_3SLOT_PATH, "XGBoost 12h")
        self._xgb_6slot = self._load_model(config.XGB_6SLOT_PATH, "XGBoost 24h")

        # ── Pre-compute SHAP background (small sample for speed) ─────────
        self._shap_background = None
        if self._xgb_3slot is not None:
            try:
                bg_idx = np.random.RandomState(42).choice(
                    len(self._df), size=min(config.SHAP_SAMPLE_SIZE, len(self._df)), replace=False
                )
                bg_data = self._df.iloc[bg_idx][self._xgb_features].copy()
                bg_data = self._apply_imputation(bg_data, mode="xgb")
                self._shap_background = bg_data
                logger.info(f"  SHAP background sample prepared: {len(bg_data)} rows")
            except Exception as e:
                logger.warning(f"  SHAP background preparation failed: {e}")

        # ── Fleet Anomaly Cache (Lazy) ────────────────────────────────────
        self._fleet_anomaly_cache = None  # Computed on first request

        # ── Schema Validation ────────────────────────────────────────────
        self._validate_schema()

        load_time_ms = (time.time() - t0) * 1000
        logger.info(f"ModelManager ready in {load_time_ms:.0f}ms | "
                     f"{len(self._server_list)} servers | "
                     f"Data freshness: {self._data_freshness:.1f}h old")

    # ─── Private Helpers ─────────────────────────────────────────────────

    def _load_json(self, path: Path) -> dict:
        """Load a JSON file. Returns empty dict on failure."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}

    def _load_dataset(self) -> pd.DataFrame:
        """Load master dataset from parquet (preferred) or CSV."""
        if config.DATASET_PATH.exists():
            df = pd.read_parquet(config.DATASET_PATH)
            logger.info(f"  Loaded dataset: {config.DATASET_PATH.name} ({len(df):,} rows)")
        elif config.DATASET_CSV_PATH.exists():
            df = pd.read_csv(config.DATASET_CSV_PATH)
            logger.info(f"  Loaded dataset: {config.DATASET_CSV_PATH.name} ({len(df):,} rows)")
        else:
            raise FileNotFoundError(f"No dataset found at {config.DATASET_PATH} or {config.DATASET_CSV_PATH}")

        # Ensure event_time_ping is datetime
        if "event_time_ping" in df.columns:
            df["event_time_ping"] = pd.to_datetime(df["event_time_ping"], errors="coerce")
        return df

    def _load_model(self, path: Path, name: str):
        """Load a joblib model. Returns None on failure (graceful degradation)."""
        if not path.exists():
            logger.warning(f"  [UNAVAILABLE] {name}: {path} not found")
            return None
        try:
            model = joblib.load(path)
            size_kb = path.stat().st_size / 1024
            logger.info(f"  Loaded {name}: {path.name} ({size_kb:.0f} KB)")
            return model
        except Exception as e:
            logger.error(f"  [CORRUPT] {name}: {e}")
            return None

    def _compute_data_freshness(self) -> float:
        """How many hours old is the most recent observation in the dataset?"""
        if "event_time_ping" in self._df.columns:
            max_time = self._df["event_time_ping"].max()
            if pd.notna(max_time):
                age = (pd.Timestamp.now() - max_time).total_seconds() / 3600
                return round(age, 1)
        return -1.0  # Unknown

    def _validate_schema(self):
        """Verify expected feature columns exist in the dataset."""
        missing_xgb = [c for c in self._xgb_features if c not in self._df.columns]
        missing_if = [c for c in self._iforest_features if c not in self._df.columns]

        if missing_xgb:
            raise SchemaValidationError(
                f"XGBoost features missing from dataset: {missing_xgb}"
            )
        if missing_if:
            raise SchemaValidationError(
                f"Isolation Forest features missing from dataset: {missing_if}"
            )
        logger.info("  Schema validation: PASSED")

    def _apply_imputation(self, df: pd.DataFrame, mode: str = "xgb") -> pd.DataFrame:
        """Apply domain-aware imputation rules from feature_order.json."""
        result = df.copy()
        hw_cols = self._imputation_rules.get("hardware_cols", [])
        hw_val = self._imputation_rules.get("hardware_impute_value", -1)
        lag_cols = self._imputation_rules.get("lag_cols", [])
        lag_val = self._imputation_rules.get("lag_impute_value", 0)

        for col in hw_cols:
            if col in result.columns:
                result[col] = result[col].fillna(hw_val).infer_objects(copy=False)
        for col in lag_cols:
            if col in result.columns:
                result[col] = result[col].fillna(lag_val).infer_objects(copy=False)
        # Fill any remaining NaN with 0
        result = result.fillna(0).infer_objects(copy=False)
        return result

    # ─── Public API ──────────────────────────────────────────────────────

    def list_all_servers(self) -> list[str]:
        """Return sorted list of all monitored server names."""
        return self._server_list

    def data_age_hours(self) -> float:
        """How many hours old is the latest data point?"""
        return self._data_freshness

    def get_staleness_warning(self) -> dict | None:
        """Return a warning dict if data is stale, else None."""
        age = self.data_age_hours()
        if age > config.STALE_DATA_THRESHOLD_HOURS:
            return {
                "warning": "DATA_STALE",
                "data_age_hours": age,
                "message": f"Dataset is {age:.0f} hours old (threshold: {config.STALE_DATA_THRESHOLD_HOURS}h). Results may not reflect current state.",
                "natural_phrasing": f"The latest telemetry is approximately {age:.0f} hours old, so predictions should be interpreted cautiously."
            }
        return None

    def get_server_row(self, server_name: str, slot: str = "latest") -> pd.Series | None:
        """
        Get a single observation row for a server.
        slot="latest" returns the most recent time slot.
        """
        mask = self._df["machine_name"] == server_name
        if not mask.any():
            return None

        server_df = self._df[mask].sort_values("event_time_ping")
        if slot == "latest":
            return server_df.iloc[-1]
        else:
            slot_mask = server_df["monitoring_slot"] == slot
            if slot_mask.any():
                return server_df[slot_mask].iloc[-1]
            return server_df.iloc[-1]

    def get_server_history(self, server_name: str, n_slots: int = 6) -> pd.DataFrame:
        """Get the last N monitoring slots for a server, sorted chronologically."""
        mask = self._df["machine_name"] == server_name
        if not mask.any():
            return pd.DataFrame()
        return self._df[mask].sort_values("event_time_ping").tail(n_slots)

    def _ensure_fleet_cache(self):
        """Lazily compute and cache fleet-wide anomaly scores on first access."""
        if self._fleet_anomaly_cache is None:
            self._fleet_anomaly_cache = self._compute_fleet_anomaly_scores()
        return self._fleet_anomaly_cache

    def _compute_fleet_anomaly_scores(self) -> dict:
        """
        Score ALL servers at once via vectorized decision_function().
        Computes threshold from contamination percentile, ranks, and percentiles.
        Called lazily on first fleet-level or enriched anomaly request.
        """
        if self._iforest is None:
            logger.warning("Cannot compute fleet scores: Isolation Forest not loaded")
            return {}

        t0 = time.perf_counter()

        # Get latest observation per server
        latest_idx = self._df.groupby("machine_name")["event_time_ping"].idxmax()
        latest_df = self._df.loc[latest_idx].copy()

        # Prepare feature matrix
        X = latest_df[self._iforest_features].copy()
        X = self._apply_imputation(X, mode="iforest")

        # Vectorized scoring — single call for all servers
        raw_scores = -self._iforest.decision_function(X)  # Negate: higher = more anomalous
        predictions = self._iforest.predict(X)

        # Compute threshold from contamination percentile
        contamination = self._thresholds.get("isolation_forest", {}).get("contamination", 0.02)
        threshold_percentile = (1.0 - contamination) * 100  # e.g., 98th for contamination=0.02
        threshold = float(np.percentile(raw_scores, threshold_percentile))

        # Build per-server results
        server_names = latest_df["machine_name"].values
        scores_array = raw_scores.astype(float)

        # Compute percentile for each server
        sorted_scores = np.sort(scores_array)
        server_scores = {}
        for i, srv in enumerate(server_names):
            score = float(scores_array[i])
            # percentileofscore: what percentage of scores are <= this score
            pct = float(np.searchsorted(sorted_scores, score, side='right') / len(sorted_scores) * 100)
            server_scores[srv] = {
                "anomaly_score": round(score, 4),
                "prediction": int(predictions[i]),
                "is_anomaly": int(predictions[i]) == -1,
                "percentile": round(pct, 1),
            }

        # Rank by score descending (rank 1 = most anomalous)
        ranked = sorted(server_scores.items(), key=lambda x: x[1]["anomaly_score"], reverse=True)
        for rank, (srv, data) in enumerate(ranked, start=1):
            data["fleet_rank"] = rank
            data["score_delta"] = round(data["anomaly_score"] - threshold, 4)

        # Fleet statistics
        total_anomalies = sum(1 for d in server_scores.values() if d["is_anomaly"])

        cache = {
            "server_scores": server_scores,
            "threshold": round(threshold, 4),
            "threshold_percentile": threshold_percentile,
            "contamination": contamination,
            "total_servers": len(server_scores),
            "total_anomalies": total_anomalies,
            "anomaly_rate_pct": round(total_anomalies / max(len(server_scores), 1) * 100, 1),
            "score_distribution": {
                "min": round(float(np.min(scores_array)), 4),
                "max": round(float(np.max(scores_array)), 4),
                "mean": round(float(np.mean(scores_array)), 4),
                "median": round(float(np.median(scores_array)), 4),
                "std": round(float(np.std(scores_array)), 4),
            },
            "all_scores_sorted": [round(float(s), 4) for s in sorted_scores],
            "ranked_servers": [(srv, data) for srv, data in ranked],
        }

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"  Fleet anomaly scores computed in {latency_ms:.0f}ms | "
                     f"{cache['total_anomalies']} anomalies / {cache['total_servers']} servers | "
                     f"threshold={cache['threshold']}")
        return cache

    def get_fleet_anomaly_stats(self) -> dict:
        """
        Return fleet-level anomaly distribution statistics.
        Lazy-computed on first call.
        """
        cache = self._ensure_fleet_cache()
        if not cache:
            return {"error": "Fleet anomaly cache unavailable (Isolation Forest not loaded)"}

        return {
            "threshold": cache["threshold"],
            "threshold_method": f"{cache['threshold_percentile']:.0f}th percentile (contamination={cache['contamination']})",
            "total_servers": cache["total_servers"],
            "total_anomalies": cache["total_anomalies"],
            "anomaly_rate_pct": cache["anomaly_rate_pct"],
            "score_distribution": cache["score_distribution"],
            "all_scores": cache["all_scores_sorted"],
        }

    def get_top_anomalies(self, n: int = 10) -> list[dict]:
        """Return the top N most anomalous servers, ranked by score."""
        cache = self._ensure_fleet_cache()
        if not cache:
            return []

        result = []
        for srv, data in cache["ranked_servers"][:n]:
            if not data["is_anomaly"]:
                break  # Stop once we're past anomalies
            result.append({
                "server_name": srv,
                "anomaly_score": data["anomaly_score"],
                "fleet_rank": data["fleet_rank"],
                "percentile": data["percentile"],
                "score_delta": data["score_delta"],
            })
        return result

    def score_iforest(self, server_name: str) -> dict:
        """
        Score a single server with the Isolation Forest anomaly detector.
        Returns enriched output with threshold, percentile, rank, and score delta
        pulled from the lazy-computed fleet cache.
        """
        if self._iforest is None:
            return {"error": "Isolation Forest model not available", "model": "isolation_forest.joblib"}

        row = self.get_server_row(server_name)
        if row is None:
            return {"error": f"Server '{server_name}' not found",
                    "available_sample": self._server_list[:5]}

        t0 = time.perf_counter()
        X = pd.DataFrame([row[self._iforest_features]])
        X = self._apply_imputation(X, mode="iforest")

        score = -float(self._iforest.decision_function(X)[0])
        prediction = int(self._iforest.predict(X)[0])
        is_anomaly = prediction == -1
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        contamination = self._thresholds.get("isolation_forest", {}).get("contamination", 0.02)

        # Enrich with fleet context from lazy cache
        cache = self._ensure_fleet_cache()
        fleet_data = cache.get("server_scores", {}).get(server_name, {}) if cache else {}

        threshold = cache.get("threshold", 0.0) if cache else 0.0
        percentile = fleet_data.get("percentile", 0.0)
        fleet_rank = fleet_data.get("fleet_rank", 0)
        score_delta = round(score - threshold, 4)
        total_servers = cache.get("total_servers", 0) if cache else 0

        return {
            "server_name": server_name,
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "isolation_forest_prediction": prediction,
            "threshold": threshold,
            "score_delta": score_delta,
            "percentile": percentile,
            "fleet_rank": fleet_rank,
            "total_fleet_servers": total_servers,
            "contamination": contamination,
            "metadata": {
                "model_name": "isolation_forest.joblib",
                "model_version": "1.0.0-phase5.2",
                "threshold_method": f"{(1-contamination)*100:.0f}th percentile",
                "prediction_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "inference_latency_ms": latency_ms
            }
        }

    def get_server_feature_vector(self, server_name: str) -> dict:
        """
        Return the imputed feature vector for a server as a human-readable dict.
        Hardware status codes are mapped to labels (OK, Warning, Critical).
        These are the 'Key Observed Signals' — the raw values fed into the model.
        """
        row = self.get_server_row(server_name)
        if row is None:
            return {"error": f"Server '{server_name}' not found"}

        X = pd.DataFrame([row[self._iforest_features]])
        X = self._apply_imputation(X, mode="iforest")
        feature_row = X.iloc[0]

        hw_status_map = {-1: "No Sensor", 0: "OK", 1: "Warning", 2: "Critical"}
        hw_cols = {col for col, _ in config.HARDWARE_SUBSYSTEMS}

        result = {}
        for feat in self._iforest_features:
            val = feature_row[feat]
            if feat in hw_cols:
                result[feat] = hw_status_map.get(int(val), f"Unknown ({val})")
            else:
                result[feat] = round(float(val), 4) if isinstance(val, (float, np.floating)) else int(val)
        return result

    def get_telemetry_sources(self, server_name: str) -> list[str]:
        """
        Return clean evidence source names based on which vendor data is present.
        Always includes 'Ping Status Monitoring'. Adds vendor-specific sources
        based on has_dell / has_hpe flags.
        """
        row = self.get_server_row(server_name)
        if row is None:
            return []

        sources = ["Ping Status Monitoring"]
        if safe_int(row.get("has_dell", 0)) == 1:
            sources.append("Dell iDRAC Health")
        if safe_int(row.get("has_hpe", 0)) == 1:
            sources.append("HPE iLO Health")
        return sources


    def score_xgb(self, server_name: str, horizon: str = "3slot") -> dict:
        """Score a single server with XGBoost failure prediction."""
        model = self._xgb_3slot if horizon == "3slot" else self._xgb_6slot
        model_name = "xgboost_failure_3slot.joblib" if horizon == "3slot" else "xgboost_failure_6slot.joblib"

        if model is None:
            return {"error": f"XGBoost model not available", "model": model_name}

        row = self.get_server_row(server_name)
        if row is None:
            return {"error": f"Server '{server_name}' not found",
                    "available_sample": self._server_list[:5]}

        t0 = time.perf_counter()
        X = pd.DataFrame([row[self._xgb_features]])
        X = self._apply_imputation(X, mode="xgb")

        prob = float(model.predict_proba(X)[:, 1][0])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        target_key = f"target_failure_{horizon}"
        optimal_f1 = self._thresholds.get(target_key, {}).get("optimal_f1_threshold", 0.5)

        return {
            "server_name": server_name,
            "failure_probability": round(prob, 4),
            "failure_probability_pct": round(prob * 100, 1),
            "horizon": "12h" if horizon == "3slot" else "24h",
            "metadata": {
                "model_name": model_name,
                "model_version": "1.0.0-phase5.6",
                "decision_threshold": optimal_f1,
                "prediction_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "inference_latency_ms": latency_ms
            }
        }

    def get_shap_explanation(self, server_name: str, horizon: str = "3slot") -> dict:
        """Compute local SHAP attribution for a specific server's prediction."""
        model = self._xgb_3slot if horizon == "3slot" else self._xgb_6slot
        if model is None:
            return {"error": "XGBoost model not available for SHAP"}

        row = self.get_server_row(server_name)
        if row is None:
            return {"error": f"Server '{server_name}' not found"}

        X = pd.DataFrame([row[self._xgb_features]])
        X = self._apply_imputation(X, mode="xgb")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_row = shap_values[0]

            contributions = []
            for i, feat in enumerate(self._xgb_features):
                contributions.append({
                    "feature": feat,
                    "value": round(float(X.iloc[0][feat]), 4),
                    "shap_contribution": round(float(shap_row[i]), 4)
                })
            # Sort by absolute SHAP contribution
            contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)

            return {
                "server_name": server_name,
                "horizon": "12h" if horizon == "3slot" else "24h",
                "top_drivers": contributions[:6],
                "positive_drivers": [c for c in contributions if c["shap_contribution"] > 0][:4],
                "protective_factors": [c for c in contributions if c["shap_contribution"] < 0][:3],
                "base_value": round(float(explainer.expected_value), 4)
            }
        except Exception as e:
            return {"error": f"SHAP computation failed: {str(e)}"}

    def get_thresholds(self) -> dict:
        """Return the full thresholds configuration."""
        return self._thresholds

    def get_feature_meta(self) -> dict:
        """Return the full feature metadata configuration."""
        return self._feature_meta

    def find_server_by_ip(self, ip_address: str) -> str | None:
        """Find hostname for a given IP address."""
        mask = self._df["ip_address"] == ip_address
        if not mask.any():
            return None
        return str(self._df[mask].iloc[-1]["machine_name"])
