import pandas as pd
import numpy as np

def analyze_transitions():
    df = pd.read_parquet('datasets/features_stage1_generic_v1.parquet')
    
    # Sort chronologically per machine
    df = df.sort_values(by=['machine_name', 'monitoring_slot'])
    
    print("=== State Transition Analysis ===")
    
    # 1. Ping Status Transitions
    print("\n--- Ping Status Transitions ---")
    df['ping_prev'] = df.groupby('machine_name')['ping_status_binary'].shift(1)
    
    ping_trans = pd.crosstab(
        df['ping_prev'].fillna(-1).astype(int), 
        df['ping_status_binary'].fillna(-1).astype(int), 
        rownames=['Previous Slot'], 
        colnames=['Current Slot']
    )
    print(ping_trans)
    
    # 2. CPU Transitions
    print("\n--- CPU Worst Status Transitions ---")
    df['cpu_prev'] = df.groupby('machine_name')['hardware_cpu_worst_status'].shift(1)
    
    # Only consider non-NaN rows for CPU
    cpu_df = df.dropna(subset=['hardware_cpu_worst_status', 'cpu_prev'])
    
    if len(cpu_df) > 0:
        cpu_trans = pd.crosstab(
            cpu_df['cpu_prev'].astype(int), 
            cpu_df['hardware_cpu_worst_status'].astype(int),
            rownames=['Previous CPU Rank'], 
            colnames=['Current CPU Rank']
        )
        print(cpu_trans)
    else:
        print("No CPU transitions found (all NaN).")

    # 3. How many servers ever change state?
    print("\n--- Server Volatility ---")
    
    # Ping volatility
    ping_changes = (df['ping_status_binary'] != df['ping_prev']) & df['ping_prev'].notna()
    ping_volatile_servers = df[ping_changes]['machine_name'].nunique()
    total_servers = df['machine_name'].nunique()
    print(f"Servers with at least one Ping state change: {ping_volatile_servers} / {total_servers}")
    
    # CPU volatility
    cpu_changes = (df['hardware_cpu_worst_status'] != df['cpu_prev']) & df['cpu_prev'].notna()
    cpu_volatile_servers = df[cpu_changes]['machine_name'].nunique()
    hardware_servers = df.dropna(subset=['hardware_cpu_worst_status'])['machine_name'].nunique()
    print(f"Servers with at least one CPU state change: {cpu_volatile_servers} / {hardware_servers}")
    
    # 4. Do failures have warning signs?
    print("\n--- Warning Signs before Critical CPU ---")
    # Find all instances where Current CPU == 3
    critical_events = cpu_df[cpu_df['hardware_cpu_worst_status'] == 3]
    print(f"Total CPU Critical observations with a previous state: {len(critical_events)}")
    if len(critical_events) > 0:
        prev_states = critical_events['cpu_prev'].value_counts().sort_index()
        for state, count in prev_states.items():
            print(f"  Preceded by Rank {int(state)}: {count} ({count/len(critical_events)*100:.1f}%)")

if __name__ == "__main__":
    analyze_transitions()
