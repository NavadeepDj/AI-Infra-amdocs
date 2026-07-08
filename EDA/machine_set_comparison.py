import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 3: Machine Set Comparison
#
# Goal:
# - Compare which machines appear in Ping, HPE iLO, and Dell iDRAC
# - Check whether hardware-monitored machines are also present in Ping
# - Check whether HPE and Dell machine lists overlap
#
# Why this matters:
# If we want to merge datasets later, we must understand whether the
# machine populations are the same or only partially overlapping.
# ==========================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "datasets"

ping_file = DATA_FOLDER / "ping_status_export_20260702_mockup.csv"
hpe_file = DATA_FOLDER / "hpe_ilo_health_export_20260702_mockup.csv"
dell_file = DATA_FOLDER / "dell_idrac_health_ext_export_20260702_mockup.csv"

ping = pd.read_csv(ping_file)
hpe = pd.read_csv(hpe_file)
dell = pd.read_csv(dell_file)

ping_set = set(ping["vm_name"].dropna().unique())
hpe_set = set(hpe["server_name"].dropna().unique())
dell_set = set(dell["server_name"].dropna().unique())

print("=" * 80)
print("STEP 3: MACHINE SET COMPARISON")
print("=" * 80)

print(f"Ping Machines : {len(ping_set)}")
print(f"HPE Machines  : {len(hpe_set)}")
print(f"Dell Machines : {len(dell_set)}")

print("\nOverlap")
print("-" * 40)
print(f"Ping and HPE  : {len(ping_set & hpe_set)}")
print(f"Ping and Dell : {len(ping_set & dell_set)}")
print(f"HPE and Dell  : {len(hpe_set & dell_set)}")
print(f"All three     : {len(ping_set & hpe_set & dell_set)}")

print("\nOnly in one source")
print("-" * 40)
print(f"Only Ping : {len(ping_set - hpe_set - dell_set)}")
print(f"Only HPE  : {len(hpe_set - ping_set - dell_set)}")
print(f"Only Dell : {len(dell_set - ping_set - hpe_set)}")

print("\nMachines in HPE and Dell")
print("-" * 40)
for machine in sorted(hpe_set & dell_set):
    print(machine)

print("\nSample machines only in Ping")
print("-" * 40)
for machine in sorted(ping_set - hpe_set - dell_set)[:20]:
    print(machine)
