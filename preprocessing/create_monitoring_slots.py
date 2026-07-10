import pandas as pd
import sys
from standardize_schema import standardize_schemas

def parse_and_create_slots():
    # --- Load Step 2 Standardized Data ---
    ping_df, hpe_df, dell_df = standardize_schemas()
    
    print("=== Step 3: Timestamp Standardization ===")
    
    # Convert 'event_time' strings to strict datetime format
    for label, df in [("Ping Status", ping_df), ("HPE iLO Health", hpe_df), ("Dell iDRAC Ext", dell_df)]:
        try:
            df["event_time"] = pd.to_datetime(df["event_time"], errors="raise", format="mixed", dayfirst=True)
        except Exception as e:
            print(f"[ERROR] Date parsing failed for {label}: {e}")
            sys.exit(1)
        print(f"[OK] Parsed timestamps for {label}.")
        print(f"     Time Horizon: {df['event_time'].min()} to {df['event_time'].max()}")
        
    print("[SUCCESS] Step 3: Timestamp standardization complete.\n")
    
    print("=== Step 4: Create Monitoring Slot ===")
    
    # Map raw event_times onto our canonical 4-hour monitoring slot grid
    for label, df in [("Ping Status", ping_df), ("HPE iLO Health", hpe_df), ("Dell iDRAC Ext", dell_df)]:
        # Vectorized allocation to nearest center hours: 02, 06, 10, 14, 18, 22
        slot_hours = ((df["event_time"].dt.hour // 4) * 4 + 2).astype(str).str.zfill(2)
        df["monitoring_slot"] = df["event_time"].dt.strftime("%Y-%m-%d") + "_Slot-" + slot_hours
        
        # Validation checks
        # 1. Null check
        if df["monitoring_slot"].isnull().any():
            print(f"[ERROR] Found null monitoring slots in {label}.")
            sys.exit(1)
            
        # 2. Uniqueness check (Duplicate keys within individual datasets)
        dup_mask = df.duplicated(subset=["machine_name", "ip_address", "monitoring_slot"], keep=False)
        dup_count = dup_mask.sum()
        if dup_count > 0:
            print(f"[ERROR] Duplicate observation keys found in {label}: {dup_count} rows.")
            print(df[dup_mask].sort_values(by=["machine_name", "monitoring_slot"]).head(10))
            sys.exit(1)
            
        print(f"[OK] {label} monitoring slots verified unique.")
        print(f"     Unique Slots Count: {df['monitoring_slot'].nunique()}")
        print(f"     Unique Machine-IP Pairs: {df.groupby(['machine_name', 'ip_address']).ngroups}")
        
    print("[SUCCESS] Step 4: Monitoring slot creation and pre-merge checks complete.\n")
    return ping_df, hpe_df, dell_df

if __name__ == "__main__":
    parse_and_create_slots()
