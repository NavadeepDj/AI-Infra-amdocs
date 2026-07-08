# Infrastructure Health Prediction System

## Phase 1: Dataset Understanding & Initial Analysis

### Objective

Before developing any Machine Learning or AI models, the first step was to understand the available datasets and determine how they relate to one another. This phase focused on analyzing the structure, purpose, and relationships between the different monitoring data sources.

---

# 1. Available Datasets

The business scenario states that infrastructure health information is collected daily from multiple monitoring systems. From the provided data, three major datasets were analyzed.

## 1.1 Ping Status Dataset

**Purpose**

The Ping Status dataset monitors whether a server or virtual machine is reachable over the network.

**Important Columns**

|Column|Description|
|---|---|
|id|Record Identifier|
|vm_name|Name of the monitored machine|
|vm_ip|IP Address|
|status|Reachable / Unreachable|
|timestamp|Time when ping was performed|

### Information Provided

This dataset only measures **network availability**.

It does **not** explain _why_ a machine is unhealthy.

It only answers:

> "Can this machine be reached over the network?"

Typical status values include:

- Reachable
    
- Unreachable
    

---

## 1.2 HPE iLO Health Dataset

**Purpose**

The HPE iLO dataset provides hardware health information for HPE servers.

**Important Columns**

|Column|Description|
|---|---|
|server_name|Server Name|
|ip_address|Server IP|
|fans|Fan Health|
|cpu|CPU Health|
|memory|Memory Health|
|storage|Storage Health|
|temperature|Temperature Status|
|power|Power Supply Health|
|recorded_at|Timestamp|
|current_problems|Active Hardware Problems|

### Information Provided

Unlike Ping Status, this dataset monitors the physical health of the infrastructure.

It answers questions such as:

- Is CPU healthy?
    
- Is memory healthy?
    
- Are fans functioning?
    
- Is the server overheating?
    
- Are there any active hardware problems?
    

---

## 1.3 Dell iDRAC Dataset

**Purpose**

The Dell iDRAC dataset performs hardware monitoring similar to HPE iLO but for Dell servers.

**Important Columns**

|Column|Description|
|---|---|
|server_name|Server Name|
|ip_address|Server IP|
|overall_status|Overall Health|
|cpu|CPU Health|
|memory|Memory Health|
|storage|Storage Health|
|fans|Fan Health|
|temperature|Temperature Health|
|power|Power Health|
|issues_detected|Detected Issues|
|comments|Additional Comments|
|current_problems|Active Problems|

### Information Provided

This dataset provides a complete overview of server hardware health and often includes more descriptive diagnostic information compared to the HPE iLO dataset.

---

# 2. Initial Assumptions

Initially, it appeared that:

- Ping Status monitored Virtual Machines
    
- HPE iLO monitored Physical Servers
    
- Dell iDRAC monitored Physical Servers
    

This raised an important architectural question:

> Does one virtual machine correspond to multiple physical servers?

This assumption required verification.

---

# 3. Relationship Investigation

To validate the relationship between datasets, a common machine was selected.

Example:

Machine Name

```
v5G-AMF-Backup-02
```

IP Address

```
100.100.58.45
```

This machine appeared in all three datasets.

### Ping Dataset

|Name|IP|
|---|---|
|v5G-AMF-Backup-02|100.100.58.45|

---

### HPE iLO

|Server|IP|
|---|---|
|v5G-AMF-Backup-02|100.100.58.45|

---

### Dell iDRAC

|Server|IP|
|---|---|
|v5G-AMF-Backup-02|100.100.58.45|

---

## Observation

Both

- Machine Name
    
- IP Address
    

matched perfectly across all three datasets.

This indicates that all three monitoring systems are observing the same infrastructure asset.

Instead of representing different machines, each dataset provides a different perspective of the same machine.

---

# 4. Understanding the Monitoring Architecture

Initially, it was assumed that the Ping dataset represented Virtual Machines while iLO and iDRAC represented their host servers.

However, after inspecting multiple records, a more accurate interpretation emerged.

The monitored entity remains the same across all datasets.

Each monitoring tool records different aspects of the same infrastructure component.

```
                 Infrastructure Machine

         Name : v5G-AMF-Backup-02
         IP   : 100.100.58.45

                    │
     ┌──────────────┼──────────────┐
     │              │              │
     ▼              ▼              ▼
 Ping Status      HPE iLO      Dell iDRAC
 Network        Hardware      Hardware
 Availability    Health        Health
```

Thus, the datasets are complementary rather than independent.

---

# 5. Time-Series Investigation

The next step was to inspect repeated records for the same machine.

Example (Ping Dataset)

|Timestamp|
|---|
|02:24|
|06:24|
|10:24|
|14:24|
|18:24|
|22:24|

Observation:

Monitoring occurs every **4 hours**.

The same investigation was performed on HPE iLO.

|Timestamp|
|---|
|02:46|
|06:46|
|10:46|
|14:46|
|18:46|
|22:46|

Again,

Monitoring occurred every **4 hours**.

Dell iDRAC showed a similar pattern.

Typical timestamps:

- 02:45
    
- 06:45
    
- 10:45
    
- 14:45
    
- 18:45
    
- 22:45
    

Extended iDRAC export showed:

- 02:48
    
- 06:48
    
- 10:48
    
- 14:48
    
- 18:48
    
- 22:48
    

---

# 6. Important Discovery

Although all monitoring systems operate every four hours, they do **not** collect data at the exact same minute.

Observed schedule:

|Monitoring System|Example Time|
|---|---|
|Ping|02:24|
|Dell iDRAC|02:45|
|HPE iLO|02:46|

The same offset repeats throughout the day.

This indicates that each monitoring tool executes independently as part of a monitoring schedule.

---

# 7. Understanding Monitoring Cycles

Instead of treating timestamps literally, it is more appropriate to view them as belonging to the same monitoring cycle.

Example

```
Monitoring Cycle

02:00 – 03:00

    Ping

        ↓

    Dell iDRAC

        ↓

    HPE iLO
```

Although timestamps differ slightly, they describe the same health snapshot of the infrastructure.

This concept becomes important while merging datasets.

---

# 8. Machine Identity

From the investigation, the following observations were made:

- Machine Name consistently maps to a single IP Address.
    
- Server Name matches Machine Name.
    
- IP Address remains consistent across monitoring systems.
    
- Multiple records exist because the same machine is monitored repeatedly over time.
    

Therefore, a single row does **not** represent a unique server.

Instead,

```
One Row

=

One Machine

+

One Monitoring Timestamp
```

This transforms the dataset into a **time-series monitoring dataset** rather than a simple inventory dataset.

---

# 9. Implications for Machine Learning

This discovery significantly influences the ML pipeline.

Instead of learning from isolated records, models can learn from infrastructure behavior over time.

Examples include:

- Temperature gradually increasing
    
- Fan degradation over several monitoring cycles
    
- Increasing ping failures
    
- Hardware warnings preceding outages
    

These temporal patterns are essential for:

- Failure Prediction
    
- Anomaly Detection
    
- Root Cause Analysis
    
- Time-Series Forecasting
    

---

# 10. Initial Merge Strategy

Since timestamps are not perfectly synchronized, merging datasets using exact timestamps is not recommended.

Instead, the proposed merge strategy is:

Primary Keys

```
Machine Name (or Server Name)

+

IP Address

+

Monitoring Cycle
```

where "Monitoring Cycle" represents the nearest scheduled monitoring window rather than the exact timestamp.

This strategy enables all monitoring information belonging to the same observation window to be combined into a single Machine Learning record.

---

# 11. Current Understanding

At the end of the dataset analysis phase, the following conclusions were reached:

- The three datasets are complementary monitoring sources.
    
- All datasets describe the same infrastructure machines.
    
- Monitoring occurs every four hours.
    
- Different monitoring tools execute a few minutes apart.
    
- Each record represents one machine at one monitoring instant.
    
- The datasets form a multivariate time-series suitable for predictive analytics.
    
- The final Machine Learning dataset should be constructed by combining all monitoring sources into unified monitoring-cycle observations.
    

---

# Next Phase

The next stage of the project will focus on:

1. Data Profiling
    
2. Data Quality Assessment
    
3. Missing Value Analysis
    
4. Duplicate Detection
    
5. Unified Dataset Construction
    
6. Feature Engineering
    

These steps will prepare the data for anomaly detection, failure prediction, forecasting, and AI-powered infrastructure health analysis.