# Step 8: Monitoring Timeline Validation

## Question

Step 7 showed that the datasets are regular at the summary level.

Now we validate the timeline at the individual machine level:

```text
Does every machine have every expected monitoring slot?
Are intervals consistently 4 hours?
Are there duplicate machine + timestamp records?
Is machine + IP + monitoring_slot unique?
```

## Script Used

[timeline_validation.py](C:/Users/navad/ML_data/EDA/timeline_validation.py)

## Files Used

We continued with the aligned `20260702` files:

```text
ping_status_export_20260702_mockup.csv
hpe_ilo_health_export_20260702_mockup.csv
dell_idrac_health_ext_export_20260702_mockup.csv
```

## Expected Timeline

The expected schedule is:

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
31 days x 6 monitoring cycles per day = 186 observations per machine
```

## Results

| Dataset | Expected Observations per Machine | Actual Min | Actual Max | Wrong Count Machines | Wrong Hour Machines | Non-4-Hour Interval Machines | Duplicate Timestamp Keys | Duplicate Slot Keys |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ping | 186 | 186 | 186 | 0 | 0 | 0 | 0 | 0 |
| HPE iLO | 186 | 186 | 186 | 0 | 0 | 0 | 0 | 0 |
| Dell iDRAC | 186 | 186 | 186 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

This is stronger than the earlier frequency check.

We have now shown that every monitored machine has:

```text
Exactly 186 observations
The same six monitoring hours every day
Consistent 4-hour monitoring intervals
No duplicate machine + IP + timestamp rows
No duplicate machine + IP + monitoring_slot rows
```

## What This Means

The aligned files are not only regular at the dataset level.

They are also regular at the individual machine level.

That means the final ML dataset can safely use:

```text
machine_name + ip_address + monitoring_slot
```

as the observation key.

## Current Conclusion

The time dimension is clean and suitable for merging.

Every row in the final merged dataset should represent:

```text
One machine
+
One monitoring cycle
=
One infrastructure health observation
```

## Next Step

We are now ready for the first actual merge design:

```text
Step 9: Build Unified Monitoring Dataset
```

The merge should:

```text
1. Standardize machine and timestamp column names
2. Create monitoring_slot in each dataset
3. Use Ping as the base inventory
4. Left join HPE iLO
5. Left join Dell iDRAC
6. Preserve source-specific hardware columns
```
