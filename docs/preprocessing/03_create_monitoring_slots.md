# Step 3 & 4: Datetime Normalization & Monitoring Slot Generation Log

**Script File:** [`preprocessing/create_monitoring_slots.py`](file:///c:/Users/navad/ML_data/preprocessing/create_monitoring_slots.py)  
**Execution Status:** `PASS`  

---

## 1. Step 3: Datetime Normalization Protocol
Raw `event_time` strings across files exhibit format variance (`DD/MM/YYYY` in Ping vs `YYYY-MM-DD` in hardware tables). We strictly normalize strings to Python `datetime64[ns]` objects using `pd.to_datetime(..., errors="raise", format="mixed", dayfirst=True)`. This resolves format variances and guarantees zero out-of-bounds month/day swaps.

### Verified Time Horizon Bounds
- **Ping Status:** `2026-06-02 02:00:00` to `2026-07-02 22:59:00`
- **HPE iLO Health:** `2026-06-02 02:02:00` to `2026-07-02 22:49:00`
- **Dell iDRAC Extended:** `2026-06-02 02:47:00` to `2026-07-02 22:50:00`

---

## 2. Step 4: Monitoring Slot Generation Algorithm
Because inter-system clock jitter causes timestamps to drift across up to 46 minutes (`02:00` vs `02:24` vs `02:46`), raw timestamps cannot be joined directly. We bucketize all observations into a canonical **4-hour monitoring cycle** (`6` cycles/day):

```python
# Mathematical formula for center-hour mapping
slot_hour = ((dt.hour // 4) * 4) + 2

# Cycle mapping window:
[00:00 - 03:59] -> Slot-02
[04:00 - 07:59] -> Slot-06
[08:00 - 11:59] -> Slot-10
[12:00 - 15:59] -> Slot-14
[16:00 - 19:59] -> Slot-18
[20:00 - 23:59] -> Slot-22
```

The resulting `monitoring_slot` string (`YYYY-MM-DD_Slot-HH`) serves as our immutable temporal join key.

---

## 3. Verified Execution Results

```text
=== Step 3: Timestamp Standardization ===
[OK] Parsed timestamps for Ping Status.
     Time Horizon: 2026-06-02 02:00:00 to 2026-07-02 22:59:00
[OK] Parsed timestamps for HPE iLO Health.
     Time Horizon: 2026-06-02 02:02:00 to 2026-07-02 22:49:00
[OK] Parsed timestamps for Dell iDRAC Ext.
     Time Horizon: 2026-06-02 02:47:00 to 2026-07-02 22:50:00
[SUCCESS] Step 3: Timestamp standardization complete.

=== Step 4: Create Monitoring Slot ===
[OK] Ping Status monitoring slots verified unique.
     Unique Slots Count: 186
     Unique Machine-IP Pairs: 246
[OK] HPE iLO Health monitoring slots verified unique.
     Unique Slots Count: 186
     Unique Machine-IP Pairs: 15
[OK] Dell iDRAC Ext monitoring slots verified unique.
     Unique Slots Count: 186
     Unique Machine-IP Pairs: 26
[SUCCESS] Step 4: Monitoring slot creation and pre-merge checks complete.
```

---

## 4. Verification Verdict
**`PASS`** — Exactly `186` unique monitoring slots generated across all 31 days (`31 * 6 = 186`). Zero duplicate `machine_name + ip_address + monitoring_slot` observation keys exist inside any individual table.
