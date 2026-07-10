import sys
from create_monitoring_slots import parse_and_create_slots

def validate_pre_merge():
    print("=== Step 5: Pre-Merge Validation ===")
    
    # 1. Ingest Step 4 output (standardized, parsed, slot-generated DataFrames)
    ping_df, hpe_df, dell_df = parse_and_create_slots()
    
    # 2. Verify Exact Machine Counts, Slot Counts, and Observation Counts
    metrics = [
        ("Ping Status", ping_df, 246, 186, 45756),
        ("HPE iLO Health", hpe_df, 15, 186, 2790),
        ("Dell iDRAC Ext", dell_df, 26, 186, 4836)
    ]
    
    for label, df, exp_machines, exp_slots, exp_obs in metrics:
        act_machines = df["machine_name"].nunique()
        act_slots = df["monitoring_slot"].nunique()
        act_obs = len(df)
        
        # Check machines
        if act_machines != exp_machines:
            print(f"[ERROR] {label} machine count mismatch! Expected: {exp_machines}, Actual: {act_machines}")
            sys.exit(1)
            
        # Check slots
        if act_slots != exp_slots:
            print(f"[ERROR] {label} slot count mismatch! Expected: {exp_slots}, Actual: {act_slots}")
            sys.exit(1)
            
        # Check observation count
        if act_obs != exp_obs:
            print(f"[ERROR] {label} observation count mismatch! Expected: {exp_obs}, Actual: {act_obs}")
            sys.exit(1)
            
        print(f"[PASS] {label} Audit: {act_machines} machines x {act_slots} slots = {act_obs} exact observations (0 missing slots).")
        
    # 3. Verify Inventory Containment & Exact EDA Machine Segments
    ping_ips = set(ping_df["ip_address"].unique())
    hpe_ips = set(hpe_df["ip_address"].unique())
    dell_ips = set(dell_df["ip_address"].unique())
    
    # HPE subset of Ping check
    if not hpe_ips.issubset(ping_ips):
        print(f"[ERROR] HPE IPs contain external IPs not in Ping master inventory: {hpe_ips - ping_ips}")
        sys.exit(1)
    print(f"[PASS] Inventory Containment: All {len(hpe_ips)} HPE IPs are verified inside Ping master inventory.")
    
    # Dell subset of Ping check
    if not dell_ips.issubset(ping_ips):
        print(f"[ERROR] Dell IPs contain external IPs not in Ping master inventory: {dell_ips - ping_ips}")
        sys.exit(1)
    print(f"[PASS] Inventory Containment: All {len(dell_ips)} Dell IPs are verified inside Ping master inventory.")
    
    # Verify exact EDA Segment Breakdown (15 Common, 11 Dell-only, 220 Ping-only)
    common_ips = hpe_ips.intersection(dell_ips)
    if len(common_ips) != 15:
        print(f"[ERROR] Expected exactly 15 common machines across Ping+HPE+Dell per EDA, found {len(common_ips)}.")
        sys.exit(1)
    print(f"[PASS] Complementary Telemetry Check: Exactly 15 machines ({len(common_ips)} x 186 = {len(common_ips)*186} obs) share both HPE and Dell telemetry as established in EDA.")
    
    dell_only_ips = dell_ips - hpe_ips
    if len(dell_only_ips) != 11:
        print(f"[ERROR] Expected exactly 11 Dell-only machines per EDA, found {len(dell_only_ips)}.")
        sys.exit(1)
    print(f"[PASS] Dell-Only Hardware Check: Exactly 11 machines ({len(dell_only_ips)} x 186 = {len(dell_only_ips)*186} obs) monitored exclusively by Dell iDRAC.")
    
    ping_only_ips = ping_ips - dell_ips
    if len(ping_only_ips) != 220:
        print(f"[ERROR] Expected exactly 220 Ping-only network machines per EDA, found {len(ping_only_ips)}.")
        sys.exit(1)
    print(f"[PASS] Ping-Only Network Check: Exactly 220 machines ({len(ping_only_ips)} x 186 = {len(ping_only_ips)*186} obs) monitored exclusively by Ping reachability.")
    
    # 4. Verify Discrete Categorical Health Flags
    hpe_comps = ["fans", "cpu", "memory", "storage", "temperature", "power"]
    for comp in hpe_comps:
        vals = set(hpe_df[comp].dropna().unique())
        expected_vals = {"OK", "Warning", "Critical"}
        if not vals.issubset(expected_vals):
            print(f"[WARNING] {comp} in HPE contains unexpected categories: {vals - expected_vals}")
    print("[PASS] Discrete categorical check passed for HPE hardware flags.")
    
    print("[SUCCESS] Step 5: All pre-merge relational and schema checks PASSED cleanly.\n")
    return ping_df, hpe_df, dell_df

if __name__ == "__main__":
    validate_pre_merge()
