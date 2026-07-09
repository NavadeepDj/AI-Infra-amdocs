"""
=============================================================================
tools.py — Deterministic Analysis Tools for the Explainable Data Understanding Agent
=============================================================================

Every function in this module performs ONE well-defined analysis task and
returns a structured dictionary. These functions NEVER print, NEVER guess,
and NEVER hallucinate. They return reproducible facts from the data.

The LLM agent calls these tools and reasons over their outputs.
=============================================================================
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Helpers (internal, not exposed as tools)
# ---------------------------------------------------------------------------

def _resolve_path(file_path: str) -> Path:
    """Resolve a file path relative to the project root or as absolute."""
    p = Path(file_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _get_time_col(df: pd.DataFrame) -> str | None:
    for col in ["timestamp", "recorded_at"]:
        if col in df.columns:
            return col
    return None


def _get_machine_col(df: pd.DataFrame) -> str | None:
    for col in ["vm_name", "server_name"]:
        if col in df.columns:
            return col
    return None


def _get_ip_col(df: pd.DataFrame) -> str | None:
    for col in ["vm_ip", "ip_address"]:
        if col in df.columns:
            return col
    return None


def _monitoring_slot(ts: pd.Series) -> pd.Series:
    parsed = _parse_time(ts)
    return (parsed - pd.Timedelta(hours=2)).dt.floor("4h") + pd.Timedelta(hours=2)


# ---------------------------------------------------------------------------
# Tool: List Available Datasets
# ---------------------------------------------------------------------------

def list_datasets() -> dict:
    """Discover all CSV files available in the datasets/ directory."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    return {
        "datasets_directory": str(DATA_DIR),
        "csv_files": [f.name for f in csv_files],
        "file_count": len(csv_files),
    }


# ---------------------------------------------------------------------------
# Tool: Dataset Overview
# ---------------------------------------------------------------------------

def dataset_overview(file_path: str) -> dict:
    """Get row count, column count, memory usage, and column names for a single CSV file."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)
    return {
        "file": p.name,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "column_names": list(df.columns),
    }


# ---------------------------------------------------------------------------
# Tool: Unique Machine Profiler
# ---------------------------------------------------------------------------

def unique_machine_profiler(file_path: str) -> dict:
    """Count unique machines and IPs in a dataset. Identify which columns hold machine names and IPs."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)
    machine_col = _get_machine_col(df)
    ip_col = _get_ip_col(df)

    result = {"file": p.name}
    if machine_col:
        machines = sorted(df[machine_col].dropna().unique().tolist())
        result["machine_column"] = machine_col
        result["unique_machines"] = len(machines)
        result["sample_machines"] = machines[:10]
    else:
        result["machine_column"] = None
        result["unique_machines"] = 0

    if ip_col:
        ips = sorted(df[ip_col].dropna().unique().tolist())
        result["ip_column"] = ip_col
        result["unique_ips"] = len(ips)
        result["sample_ips"] = ips[:5]
    else:
        result["ip_column"] = None
        result["unique_ips"] = 0

    return result


# ---------------------------------------------------------------------------
# Tool: Machine Set Comparison
# ---------------------------------------------------------------------------

def machine_set_comparison() -> dict:
    """Compare which machines appear across Ping, HPE iLO, and Dell iDRAC datasets."""
    sets = {}
    for f in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        mc = _get_machine_col(df)
        if mc:
            sets[f.name] = set(df[mc].dropna().unique())

    names = list(sets.keys())
    result = {
        "datasets": {name: len(s) for name, s in sets.items()},
        "overlaps": {},
    }

    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            overlap = sets[n1] & sets[n2]
            result["overlaps"][f"{n1} ∩ {n2}"] = {
                "count": len(overlap),
                "sample": sorted(overlap)[:5],
            }

    all_sets = list(sets.values())
    if len(all_sets) >= 2:
        common_all = all_sets[0]
        for s in all_sets[1:]:
            common_all = common_all & s
        result["common_across_all"] = {
            "count": len(common_all),
            "machines": sorted(common_all)[:10],
        }

    return result


# ---------------------------------------------------------------------------
# Tool: Machine-IP Relationship Profiler
# ---------------------------------------------------------------------------

def machine_ip_relationship(file_path: str) -> dict:
    """Check whether machine names map 1-to-1 to IP addresses."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)
    mc = _get_machine_col(df)
    ic = _get_ip_col(df)

    if not mc or not ic:
        return {"file": p.name, "error": "Could not find machine or IP columns."}

    m_to_ip = df.groupby(mc)[ic].nunique()
    ip_to_m = df.groupby(ic)[mc].nunique()

    multi_ip_machines = m_to_ip[m_to_ip > 1]
    multi_m_ips = ip_to_m[ip_to_m > 1]

    mapping_type = "one-to-one" if len(multi_ip_machines) == 0 and len(multi_m_ips) == 0 else "one-to-many"

    return {
        "file": p.name,
        "machine_column": mc,
        "ip_column": ic,
        "unique_machines": int(df[mc].nunique()),
        "unique_ips": int(df[ic].nunique()),
        "mapping_type": mapping_type,
        "machines_with_multiple_ips": len(multi_ip_machines),
        "ips_with_multiple_machines": len(multi_m_ips),
        "anomaly_examples": {str(k): int(v) for k, v in multi_ip_machines.head(5).items()},
    }


# ---------------------------------------------------------------------------
# Tool: Time Range Profiler
# ---------------------------------------------------------------------------

def time_range_profiler(file_path: str) -> dict:
    """Find the earliest and latest timestamps and calculate time span."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)
    tc = _get_time_col(df)

    if not tc:
        return {"file": p.name, "error": "No timestamp column found."}

    parsed = _parse_time(df[tc])
    invalid = int(parsed.isna().sum())

    return {
        "file": p.name,
        "time_column": tc,
        "start": str(parsed.min()),
        "end": str(parsed.max()),
        "duration_days": round((parsed.max() - parsed.min()).total_seconds() / 86400, 2),
        "unique_dates": int(parsed.dt.date.nunique()),
        "invalid_timestamps": invalid,
        "sample_timestamps": [str(t) for t in parsed.dropna().head(3)],
    }


# ---------------------------------------------------------------------------
# Tool: Monitoring Frequency Analyzer
# ---------------------------------------------------------------------------

def monitoring_frequency(file_path: str) -> dict:
    """Analyze monitoring frequency: observations per machine, slot hours, intervals."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)
    mc = _get_machine_col(df)
    tc = _get_time_col(df)

    if not mc or not tc:
        return {"file": p.name, "error": "Missing machine or time column."}

    df["_slot"] = _monitoring_slot(df[tc])
    obs_per_machine = df.groupby(mc).size()
    slot_hours = sorted([int(x) for x in df["_slot"].dt.hour.dropna().unique()])

    return {
        "file": p.name,
        "unique_slots": int(df["_slot"].nunique()),
        "slot_hours": slot_hours,
        "slots_per_day": len(slot_hours),
        "obs_per_machine_min": int(obs_per_machine.min()),
        "obs_per_machine_max": int(obs_per_machine.max()),
        "obs_per_machine_median": float(obs_per_machine.median()),
        "all_machines_equal_observations": bool(obs_per_machine.min() == obs_per_machine.max()),
    }


# ---------------------------------------------------------------------------
# Tool: Column Profiler
# ---------------------------------------------------------------------------

def column_profiler(file_path: str) -> dict:
    """Profile every column: data type, missing values, unique counts, sample values."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)

    profiles = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))
        samples = sorted(df[col].dropna().astype(str).unique().tolist())[:8]

        profiles.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": missing,
            "missing_pct": round(missing / len(df) * 100, 2),
            "unique_count": unique,
            "sample_values": samples,
        })

    return {"file": p.name, "rows": len(df), "profiles": profiles}


# ---------------------------------------------------------------------------
# Tool: Timeline Validator
# ---------------------------------------------------------------------------

def timeline_validator(file_path: str) -> dict:
    """Check for duplicate timestamps, duplicate monitoring slots, and interval consistency."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)
    mc = _get_machine_col(df)
    ic = _get_ip_col(df)
    tc = _get_time_col(df)

    if not mc or not tc:
        return {"file": p.name, "error": "Missing required columns."}

    df["_time"] = _parse_time(df[tc])
    df["_slot"] = _monitoring_slot(df[tc])

    dup_ts = int(df.duplicated(subset=[mc, tc]).sum()) if ic is None else int(df.duplicated(subset=[mc, ic, "_time"]).sum())
    dup_slot = int(df.duplicated(subset=[mc, "_slot"]).sum()) if ic is None else int(df.duplicated(subset=[mc, ic, "_slot"]).sum())

    # Check interval consistency per machine
    bad_machines = []
    for machine, mdf in df.sort_values("_time").groupby(mc):
        intervals = mdf["_slot"].diff().dropna()
        hours = sorted(set(i / pd.Timedelta(hours=1) for i in intervals))
        if hours != [4.0]:
            bad_machines.append({"machine": machine, "observed_intervals_hours": hours[:5]})

    return {
        "file": p.name,
        "duplicate_timestamps": dup_ts,
        "duplicate_monitoring_slots": dup_slot,
        "machines_with_irregular_intervals": len(bad_machines),
        "irregular_examples": bad_machines[:3],
    }


# ---------------------------------------------------------------------------
# Tool: HPE vs Dell Redundancy Check
# ---------------------------------------------------------------------------

def hpe_dell_redundancy_check() -> dict:
    """Compare HPE iLO and Dell iDRAC health readings for overlapping machines."""
    hpe_files = sorted(DATA_DIR.glob("hpe_ilo_health_export_20260702*.csv"))
    dell_files = sorted(DATA_DIR.glob("dell_idrac_health_ext_export_20260702*.csv"))

    if not hpe_files or not dell_files:
        return {"error": "Could not find aligned HPE and Dell 20260702 files."}

    hpe = pd.read_csv(hpe_files[0])
    dell = pd.read_csv(dell_files[0])

    components = ["cpu", "memory", "temperature", "power", "fans", "storage"]
    components = [c for c in components if c in hpe.columns and c in dell.columns]

    hpe["_slot"] = _monitoring_slot(hpe["recorded_at"])
    dell["_slot"] = _monitoring_slot(dell["timestamp"])

    merged = hpe.merge(dell, left_on=["server_name", "ip_address", "_slot"],
                       right_on=["server_name", "ip_address", "_slot"],
                       how="inner", suffixes=("_hpe", "_dell"))

    match_rates = {}
    for comp in components:
        hc, dc = f"{comp}_hpe", f"{comp}_dell"
        if hc in merged.columns and dc in merged.columns:
            match = int((merged[hc] == merged[dc]).sum())
            match_rates[comp] = round(match / len(merged) * 100, 2) if len(merged) > 0 else 0.0

    return {
        "hpe_file": hpe_files[0].name,
        "dell_file": dell_files[0].name,
        "hpe_rows": len(hpe),
        "dell_rows": len(dell),
        "aligned_rows": len(merged),
        "component_match_rates": match_rates,
    }


# ---------------------------------------------------------------------------
# Tool: Value Distribution
# ---------------------------------------------------------------------------

def value_distribution(file_path: str, column: str) -> dict:
    """Get value counts and class balance for a specific column."""
    p = _resolve_path(file_path)
    df = pd.read_csv(p)

    if column not in df.columns:
        return {"file": p.name, "column": column, "error": f"Column '{column}' not found."}

    counts = df[column].value_counts(dropna=False)
    pcts = (counts / len(df) * 100).round(2)

    return {
        "file": p.name,
        "column": column,
        "total_rows": len(df),
        "unique_values": int(df[column].nunique(dropna=False)),
        "null_count": int(df[column].isna().sum()),
        "distribution": {str(k): {"count": int(v), "pct": float(pcts[k])} for k, v in counts.head(10).items()},
    }


# ---------------------------------------------------------------------------
# Tool: Cross-Source Consistency Checker
# ---------------------------------------------------------------------------

def cross_source_consistency(machine_name: str) -> dict:
    """For a specific machine, check its status across all monitoring sources in the same time window."""
    results = {}
    for f in sorted(DATA_DIR.glob("*20260702*.csv")):
        df = pd.read_csv(f)
        mc = _get_machine_col(df)
        if not mc:
            continue
        machine_rows = df[df[mc] == machine_name]
        if len(machine_rows) == 0:
            results[f.name] = {"found": False}
            continue

        tc = _get_time_col(df)
        info = {"found": True, "rows": len(machine_rows)}

        if "status" in df.columns:
            info["status_values"] = machine_rows["status"].value_counts().to_dict()
        if "overall_status" in df.columns:
            info["overall_status_values"] = machine_rows["overall_status"].value_counts().to_dict()
        if "current_problems" in df.columns:
            problems = machine_rows["current_problems"].dropna().unique().tolist()
            info["current_problems"] = problems[:5]

        results[f.name] = info

    return {"machine_name": machine_name, "sources": results}


# ---------------------------------------------------------------------------
# Tool Registry — Used by the Agent to know what's available
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "list_datasets": {
        "function": list_datasets,
        "description": "Discover all CSV files in the datasets/ directory. Call this first to see what data is available.",
        "parameters": {},
    },
    "dataset_overview": {
        "function": dataset_overview,
        "description": "Get row count, column count, memory, and column names for a single CSV file.",
        "parameters": {"file_path": "Path to CSV file (relative to project root or absolute)"},
    },
    "unique_machine_profiler": {
        "function": unique_machine_profiler,
        "description": "Count unique machines and IPs in a dataset. Shows which columns hold machine identifiers.",
        "parameters": {"file_path": "Path to CSV file"},
    },
    "machine_set_comparison": {
        "function": machine_set_comparison,
        "description": "Compare which machines appear across all available datasets. Shows overlaps and unique sets.",
        "parameters": {},
    },
    "machine_ip_relationship": {
        "function": machine_ip_relationship,
        "description": "Check whether machine names map 1-to-1 to IP addresses. Detects identity anomalies.",
        "parameters": {"file_path": "Path to CSV file"},
    },
    "time_range_profiler": {
        "function": time_range_profiler,
        "description": "Find earliest/latest timestamps and calculate time span for a dataset.",
        "parameters": {"file_path": "Path to CSV file"},
    },
    "monitoring_frequency": {
        "function": monitoring_frequency,
        "description": "Analyze monitoring interval: how often each machine is observed, monitoring slot hours, consistency.",
        "parameters": {"file_path": "Path to CSV file"},
    },
    "column_profiler": {
        "function": column_profiler,
        "description": "Profile every column: data type, missing values, unique counts, and sample values.",
        "parameters": {"file_path": "Path to CSV file"},
    },
    "timeline_validator": {
        "function": timeline_validator,
        "description": "Check for duplicate timestamps, duplicate monitoring slots, and interval regularity per machine.",
        "parameters": {"file_path": "Path to CSV file"},
    },
    "hpe_dell_redundancy_check": {
        "function": hpe_dell_redundancy_check,
        "description": "Compare HPE iLO and Dell iDRAC health readings for overlapping machines. Shows component match rates.",
        "parameters": {},
    },
    "value_distribution": {
        "function": value_distribution,
        "description": "Get value counts and class balance for a specific column in a dataset.",
        "parameters": {"file_path": "Path to CSV file", "column": "Column name to analyze"},
    },
    "cross_source_consistency": {
        "function": cross_source_consistency,
        "description": "For a specific machine, check its status across all monitoring sources (Ping, HPE, Dell).",
        "parameters": {"machine_name": "Exact machine/server name to look up"},
    },
}
