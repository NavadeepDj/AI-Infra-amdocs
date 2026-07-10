# Step 1: Input Validation Specification & Verification Log

**Script File:** [`preprocessing/validate_inputs.py`](file:///c:/Users/navad/ML_data/preprocessing/validate_inputs.py)  
**Execution Status:** `PASS`  

---

## 1. Objective & Engineering Rationale
Before schema transformations or joins occur, the pipeline must verify that all raw CSV inputs exist, match our aligned `20260702` 31-day operational window exact row counts, and have not suffered from upstream schema drift (missing columns or unexpected new columns). If any verification fails, the pipeline halts immediately (`sys.exit(1)`).

---

## 2. Validation Audit Protocol
The script checks three distinct requirements for every file:
1. **File Existence:** Verifies that the exact canonical file exists inside `datasets/`.
2. **Row Count Verification:** Compares `len(df)` against exact numbers established during EDA:
   - `ping_status_export_20260702_mockup.csv` $\rightarrow$ **Expected: 45,756 rows**
   - `hpe_ilo_health_export_20260702_mockup.csv` $\rightarrow$ **Expected: 2,790 rows**
   - `dell_idrac_health_ext_export_20260702_mockup.csv` $\rightarrow$ **Expected: 4,836 rows**
3. **Exact Column Check:** Verifies both that all expected columns are present (`Missing Columns Check`) and checks for schema drift (`Unexpected Columns Check`).

---

## 3. Verified Execution Results

```text
=== Step 1: Input Validation ===
[OK] Found file: Ping Status (ping_status_export_20260702_mockup.csv)
[OK] Found file: HPE iLO Health (hpe_ilo_health_export_20260702_mockup.csv)
[OK] Found file: Dell iDRAC Extended (dell_idrac_health_ext_export_20260702_mockup.csv)
[OK] Ping Status row count matches expected: 45756
[OK] HPE iLO Health row count matches expected: 2790
[OK] Dell iDRAC Extended row count matches expected: 4836
[OK] Ping Status schema contains all expected columns.
[OK] HPE iLO Health schema contains all expected columns.
[OK] Dell iDRAC Extended schema contains all expected columns.
[SUCCESS] Step 1: All raw inputs validated successfully.
```

---

## 4. Verification Verdict
**`PASS`** — All three raw CSV files exist, row counts match the 31-day `186-slot` expectation exactly, and zero missing or unexpected columns were found.
