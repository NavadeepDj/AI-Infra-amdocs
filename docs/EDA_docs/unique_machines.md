Now **this is interesting**. This is exactly why we profile data before making assumptions.

Let's analyze this like a data scientist.

---

# Step 1.2 Analysis — Unique Machines

## Summary

| Dataset     | Unique Machines |
| ----------- | --------------: |
| Ping Status |         **246** |
| HPE iLO     |          **15** |
| Dell iDRAC  |          **26** |

Immediately, something stands out.

```text
Ping
246 Machines

↓

Dell
26 Machines

↓

HPE
15 Machines
```

This is a **huge difference**.

---

# Our first hypothesis was WRONG ❌

Earlier, we thought:

> "Ping, iLO, and iDRAC are monitoring the same machines."

The data now tells us:

**Not exactly.**

Only **some** machines overlap.

For example

```text
v5G-AMF-Backup-02
```

exists in

* ✅ Ping
* ✅ HPE
* ✅ Dell

But Ping has many more machines like

```text
v5G-AMF-01
v5G-AUSF-01
...
```

that don't necessarily appear in the hardware datasets.

---

# New understanding of the infrastructure

I now think the infrastructure looks more like this.

```text
                  Entire Infrastructure

                 246 Network Endpoints
                        (Ping)

                         │
      ┌──────────────────┴──────────────────┐
      │                                     │
      ▼                                     ▼

 26 Dell Servers                     15 HPE Servers

      │                                     │
  iDRAC Monitoring                    iLO Monitoring
```

This is much more realistic.

---

# What does this mean?

Ping is **network-wide monitoring**.

It checks every important endpoint.

This includes:

* VMs
* Containers
* Appliances
* Network Functions
* Physical Servers

Basically anything that has an IP address.

---

Meanwhile,

iLO only exists on

```text
HPE Hardware
```

---

iDRAC only exists on

```text
Dell Hardware
```

---

# This explains the row counts!

Earlier we asked

> Why does Ping have 45,756 rows?

Now we know.

Because

```text
246 Machines

×

Many Monitoring Cycles

↓

45,756 observations
```

Whereas

```text
15 Machines

×

Many Monitoring Cycles

↓

2,790 observations
```

Makes perfect sense.

---

# Another observation

Look at the first machine names.

Ping

```
v5G-AMF-01
```

HPE

Starts from

```
v5G-AMF-Backup-02
```

That means

not every Ping endpoint has hardware monitoring.

Exactly what we'd expect.

---

# This changes Question 1

Remember

> How would you combine the datasets?

Now the answer becomes much smarter.

Instead of saying

> Merge everything.

We say

> Perform a **left join** using the Ping dataset as the primary monitoring dataset because it contains the complete infrastructure inventory. Enrich records with HPE iLO and Dell iDRAC data where matching hardware information exists.

That's a much more production-ready approach.

---

# One thing I want to verify

Before we finalize this conclusion, I want to know:

**Do the 15 HPE machines also exist in the 26 Dell machines?**

Because in reality, a physical server can't be both HPE and Dell.

So there are two possibilities:

### Possibility 1

The mock dataset intentionally duplicated them.

or

### Possibility 2

The names overlap, but the IPs differ.

or

### Possibility 3 (most likely)

The mock data reused names across vendors for simplicity.

We need to verify that before designing the merge.

---

# This is the next script I want you to run

Let's compare the machine sets directly.

```python
import pandas as pd

# Load datasets
ping = pd.read_csv("datasets/ping_status_export_20260702_mockup.csv")
hpe = pd.read_csv("datasets/hpe_ilo_health_export_20260702_mockup.csv")
dell = pd.read_csv("datasets/dell_idrac_health_ext_export_20260702_mockup.csv")

ping_set = set(ping["vm_name"].unique())
hpe_set = set(hpe["server_name"].unique())
dell_set = set(dell["server_name"].unique())

print("="*80)
print("MACHINE SET COMPARISON")
print("="*80)

print(f"Ping Machines : {len(ping_set)}")
print(f"HPE Machines  : {len(hpe_set)}")
print(f"Dell Machines : {len(dell_set)}")

print("\nOverlap")
print("-"*40)

print("Ping ∩ HPE :", len(ping_set & hpe_set))
print("Ping ∩ Dell:", len(ping_set & dell_set))
print("HPE ∩ Dell :", len(hpe_set & dell_set))

print("\nMachines only in Ping:")
print(len(ping_set - hpe_set - dell_set))

print("\nMachines only in HPE:")
print(len(hpe_set - ping_set))

print("\nMachines only in Dell:")
print(len(dell_set - ping_set))
```

---

## One more suggestion

I'd also make one small improvement to our profiling notebook.

Instead of just printing numbers, let's **write down every hypothesis and whether the data supports it**.

For example:

| Hypothesis                                         | Status      | Evidence                                           |
| -------------------------------------------------- | ----------- | -------------------------------------------------- |
| Every monitoring system observes the same machines | ❌ Rejected  | Ping has 246 machines; HPE has 15; Dell has 26     |
| Machine names are consistent within a dataset      | ✅ Supported | Same machine names repeat across monitoring cycles |
| Monitoring is time-series based                    | ✅ Supported | Repeated observations every ~4 hours               |

This makes your EDA read like a real investigation rather than a collection of statistics, and it will greatly strengthen your final report.

C:\Users\navad\ML_data> & C:\Users\navad\AppData\Local\Programs\Python\Python311\python.exe c:/Users/navad/ML_data/EDA/unique_machines.py
================================================================================
STEP 2: UNIQUE MACHINES
================================================================================

Dataset : dell_idrac_health_export_20260703_mockup.csv
------------------------------------------------------------
Machine Column : server_name
Unique Machines: 26

First 10 Machines:
  • v5G-AMF-Backup-02
  • v5G-AMF-Central-28
  • v5G-AMF-Core-26
  • v5G-AMF-South-18
  • v5G-AMF-West-18
  • v5G-AMF-Zone2-29
  • v5G-AUSF-Backup-17
  • v5G-AUSF-Core-01
  • v5G-AUSF-Core-13
  • v5G-AUSF-Edge-02
------------------------------------------------------------

Dataset : dell_idrac_health_ext_export_20260702_mockup.csv
------------------------------------------------------------
Machine Column : server_name
Unique Machines: 26

First 10 Machines:
  • v5G-AMF-Backup-02
  • v5G-AMF-Central-28
  • v5G-AMF-Core-26
  • v5G-AMF-South-18
  • v5G-AMF-West-18
  • v5G-AMF-Zone2-29
  • v5G-AUSF-Backup-17
  • v5G-AUSF-Core-01
  • v5G-AUSF-Core-13
  • v5G-AUSF-Edge-02
------------------------------------------------------------

Dataset : dell_idrac_health_ext_export_20260703_mockup.csv
------------------------------------------------------------
Machine Column : server_name
Unique Machines: 26

First 10 Machines:
  • v5G-AMF-Backup-02
  • v5G-AMF-Central-28
  • v5G-AMF-Core-26
  • v5G-AMF-South-18
  • v5G-AMF-West-18
  • v5G-AMF-Zone2-29
  • v5G-AUSF-Backup-17
  • v5G-AUSF-Core-01
  • v5G-AUSF-Core-13
  • v5G-AUSF-Edge-02
------------------------------------------------------------

Dataset : hpe_ilo_health_export_20260702_mockup.csv
------------------------------------------------------------
Machine Column : server_name
Unique Machines: 15

First 10 Machines:
  • v5G-AMF-Backup-02
  • v5G-AMF-Central-28
  • v5G-AMF-Core-26
  • v5G-AMF-South-18
  • v5G-AMF-West-18
  • v5G-AMF-Zone2-29
  • v5G-AUSF-Backup-17
  • v5G-AUSF-Core-01
  • v5G-AUSF-Core-13
  • v5G-AUSF-Edge-02
------------------------------------------------------------

Dataset : hpe_ilo_health_export_20260703_mockup.csv
------------------------------------------------------------
Machine Column : server_name
Unique Machines: 15

First 10 Machines:
  • v5G-AMF-Backup-02
  • v5G-AMF-Central-28
  • v5G-AMF-Core-26
  • v5G-AMF-South-18
  • v5G-AMF-West-18
  • v5G-AMF-Zone2-29
  • v5G-AUSF-Backup-17
  • v5G-AUSF-Core-01
  • v5G-AUSF-Core-13
  • v5G-AUSF-Edge-02
------------------------------------------------------------

Dataset : ping_status_export_20260702_mockup.csv
------------------------------------------------------------
Machine Column : vm_name
Unique Machines: 246

First 10 Machines:
  • v5G-AMF-01
  • v5G-AMF-Backup-02
  • v5G-AMF-Central-28
  • v5G-AMF-Core-26
  • v5G-AMF-South-18
  • v5G-AMF-West-18
  • v5G-AMF-Zone2-29
  • v5G-AUSF-01
  • v5G-AUSF-Backup-17
  • v5G-AUSF-Core-01
------------------------------------------------------------

Dataset : ping_status_export_20260703_mockup.csv
------------------------------------------------------------
Machine Column : vm_name
Unique Machines: 246

First 10 Machines:
  • v5G-AMF-01
  • v5G-AMF-Backup-02
  • v5G-AMF-Central-28
  • v5G-AMF-Core-26
  • v5G-AMF-South-18
  • v5G-AMF-West-18
  • v5G-AMF-Zone2-29
  • v5G-AUSF-01
  • v5G-AUSF-Backup-17
  • v5G-AUSF-Core-01
------------------------------------------------------------
PS C:\Users\navad\ML_data> 