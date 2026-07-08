import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 4: Machine to IP Relationship
#
# Goal:
# - Check whether each machine always maps to one IP address
# - Check whether each IP address always maps to one machine
# - Check whether overlapping HPE and Dell machines have the same IP
#
# Why this matters:
# Before merging datasets, we need to know whether machine_name and
# ip_address are stable identifiers.
# ==========================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"

datasets = [
    {
        "name": "Ping",
        "file": DATA_FOLDER / "ping_status_export_20260702_mockup.csv",
        "machine_col": "vm_name",
        "ip_col": "vm_ip",
    },
    {
        "name": "HPE iLO",
        "file": DATA_FOLDER / "hpe_ilo_health_export_20260702_mockup.csv",
        "machine_col": "server_name",
        "ip_col": "ip_address",
    },
    {
        "name": "Dell iDRAC",
        "file": DATA_FOLDER / "dell_idrac_health_ext_export_20260702_mockup.csv",
        "machine_col": "server_name",
        "ip_col": "ip_address",
    },
]


def print_mapping_quality(label, df, machine_col, ip_col):
    machine_to_ip_counts = df.groupby(machine_col)[ip_col].nunique()
    ip_to_machine_counts = df.groupby(ip_col)[machine_col].nunique()

    machines_with_multiple_ips = machine_to_ip_counts[machine_to_ip_counts > 1]
    ips_with_multiple_machines = ip_to_machine_counts[ip_to_machine_counts > 1]

    print(f"\nDataset: {label}")
    print("-" * 60)
    print(f"Unique machines                  : {df[machine_col].nunique()}")
    print(f"Unique IPs                       : {df[ip_col].nunique()}")
    print(f"Machines with more than one IP   : {len(machines_with_multiple_ips)}")
    print(f"IPs with more than one machine   : {len(ips_with_multiple_machines)}")

    if len(machines_with_multiple_ips) > 0:
        print("\nExamples: machines with multiple IPs")
        for machine in machines_with_multiple_ips.index[:10]:
            ips = sorted(df.loc[df[machine_col] == machine, ip_col].unique())
            print(f"{machine}: {ips}")

    if len(ips_with_multiple_machines) > 0:
        print("\nExamples: IPs with multiple machines")
        for ip_address in ips_with_multiple_machines.index[:10]:
            machines = sorted(df.loc[df[ip_col] == ip_address, machine_col].unique())
            print(f"{ip_address}: {machines}")


print("=" * 80)
print("STEP 4: MACHINE TO IP RELATIONSHIP")
print("=" * 80)

loaded = {}

for config in datasets:
    df = pd.read_csv(config["file"])
    loaded[config["name"]] = df
    print_mapping_quality(
        config["name"],
        df,
        config["machine_col"],
        config["ip_col"],
    )

ping_pairs = set(
    zip(loaded["Ping"]["vm_name"], loaded["Ping"]["vm_ip"])
)
hpe_pairs = set(
    zip(loaded["HPE iLO"]["server_name"], loaded["HPE iLO"]["ip_address"])
)
dell_pairs = set(
    zip(loaded["Dell iDRAC"]["server_name"], loaded["Dell iDRAC"]["ip_address"])
)

print("\nCross-dataset pair overlap")
print("-" * 60)
print(f"Ping and HPE machine/IP pairs  : {len(ping_pairs & hpe_pairs)}")
print(f"Ping and Dell machine/IP pairs : {len(ping_pairs & dell_pairs)}")
print(f"HPE and Dell machine/IP pairs  : {len(hpe_pairs & dell_pairs)}")
print(f"All three machine/IP pairs     : {len(ping_pairs & hpe_pairs & dell_pairs)}")

hpe_map = (
    loaded["HPE iLO"][["server_name", "ip_address"]]
    .drop_duplicates()
    .set_index("server_name")["ip_address"]
    .to_dict()
)
dell_map = (
    loaded["Dell iDRAC"][["server_name", "ip_address"]]
    .drop_duplicates()
    .set_index("server_name")["ip_address"]
    .to_dict()
)

overlapping_machines = sorted(set(hpe_map) & set(dell_map))
same_ip_count = sum(hpe_map[machine] == dell_map[machine] for machine in overlapping_machines)

print("\nHPE vs Dell overlapping machines")
print("-" * 60)
print(f"Overlapping machines            : {len(overlapping_machines)}")
print(f"Same IP in both datasets         : {same_ip_count}")
print(f"Different IP in both datasets    : {len(overlapping_machines) - same_ip_count}")

if same_ip_count != len(overlapping_machines):
    print("\nExamples: HPE/Dell IP mismatch")
    for machine in overlapping_machines:
        if hpe_map[machine] != dell_map[machine]:
            print(f"{machine}: HPE={hpe_map[machine]}, Dell={dell_map[machine]}")
