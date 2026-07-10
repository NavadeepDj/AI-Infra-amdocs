import pandas as pd
from validate_inputs import validate_files

def standardize_schemas():
    print("=== Step 2: Schema Standardization ===")
    
    # 1. Load validated raw datasets
    ping_df, hpe_df, dell_df = validate_files()
    
    # 2. Standardize Ping Status columns and select features
    ping_mapping = {
        "vm_name": "machine_name",
        "vm_ip": "ip_address",
        "timestamp": "event_time"
    }
    ping_std = ping_df.rename(columns=ping_mapping)
    ping_std = ping_std[["machine_name", "ip_address", "event_time", "status"]]
    print("[OK] Ping Status schema standardized.")
    print(f"     Columns: {list(ping_std.columns)}")
    print(f"     Shape: {ping_std.shape}")
    
    # 3. Standardize HPE iLO Health columns and select features
    hpe_mapping = {
        "server_name": "machine_name",
        "recorded_at": "event_time"
    }
    hpe_std = hpe_df.rename(columns=hpe_mapping)
    hpe_std = hpe_std[[
        "machine_name", "ip_address", "event_time", 
        "fans", "cpu", "memory", "storage", "temperature", "power", 
        "current_problems"
    ]]
    print("[OK] HPE iLO Health schema standardized.")
    print(f"     Columns: {list(hpe_std.columns)}")
    print(f"     Shape: {hpe_std.shape}")
    
    # 4. Standardize Dell iDRAC Health Extended columns and select features
    dell_mapping = {
        "server_name": "machine_name",
        "timestamp": "event_time"
    }
    dell_std = dell_df.rename(columns=dell_mapping)
    dell_std = dell_std[[
        "machine_name", "ip_address", "event_time", 
        "status", "overall_status", 
        "fans", "cpu", "memory", "storage", "temperature", "power", 
        "issues_detected"
    ]]
    print("[OK] Dell iDRAC Health Extended schema standardized.")
    print(f"     Columns: {list(dell_std.columns)}")
    print(f"     Shape: {dell_std.shape}")
    
    print("[SUCCESS] Step 2: Schema standardization complete.\n")
    return ping_std, hpe_std, dell_std

if __name__ == "__main__":
    standardize_schemas()
