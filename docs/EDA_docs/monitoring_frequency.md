# Step 7: Monitoring Frequency

## Question

Now that we know the aligned `20260702` files cover the same one-month period, we need to answer:

```text
How often is each machine monitored?
Do all machines follow the same monitoring schedule?
Do we really have 6 observations per day?
```

## Script Used

[monitoring_frequency.py](C:/Users/navad/ML_data/EDA/monitoring_frequency.py)

## Files Used

For this step, we used only the aligned `20260702` files:

```text
ping_status_export_20260702_mockup.csv
hpe_ilo_health_export_20260702_mockup.csv
dell_idrac_health_ext_export_20260702_mockup.csv
```

These cover the same time range:

```text
2026-06-02 to 2026-07-02
```

## Results

| Dataset | Machines | Dates | Monitoring Slots | Observations per Machine | Observations per Machine per Day |
|---|---:|---:|---:|---:|---:|
| Ping | 246 | 31 | 186 | 186 | 6 |
| HPE iLO | 15 | 31 | 186 | 186 | 6 |
| Dell iDRAC | 26 | 31 | 186 | 186 | 6 |

## Monitoring Hours

All three datasets follow the same monitoring hours:

```text
02:00
06:00
10:00
14:00
18:00
22:00
```

That means:

```text
24 hours / 4 hours = 6 monitoring cycles per day
```

## Interpretation

This confirms that the aligned `20260702` files have a very regular time-series structure.

Every machine has:

```text
31 days x 6 observations per day = 186 observations
```

So for the aligned files, there are no obvious missing observations at this frequency-check level.

## What About the 20260703 Files?

The `20260703` files should not be ignored, but we should not mix them into the first merge yet.

They are better treated as additional exports or follow-up windows.

There are three possible uses:

1. **Validation data**

   Build the logic using the clean aligned `20260702` files, then test whether the same logic works on the `20260703` files.

2. **Additional historical data**

   If date parsing and overlap are cleaned carefully, the `20260703` files can extend the training history.

3. **Export consistency check**

   Use them to verify whether daily exports are produced consistently and whether the same machines, columns, and monitoring cycles appear over time.

For learning, the best approach is:

```text
First use 20260702 files to understand the full pipeline.
Then apply the same pipeline to 20260703 files as a second pass.
```

## Current Conclusion

The monitoring frequency is consistent:

```text
Every 4 hours
6 observations per machine per day
186 observations per machine over 31 days
```

This strongly supports the idea of creating a derived time key:

```text
monitoring_slot
```

## Next Step

The next investigation should be:

```text
Step 8: Missing Monitoring Cycles
```

Even though the frequency looks perfect at a high level, we still need to verify:

```text
Does every machine have every expected monitoring slot?
Are there any machine + timestamp gaps?
```
