# Step 6: Time Range Analysis

## Question

Before building time-series features, we need to understand:

```text
What time period does each file cover?
Does each file contain approximately one month of data?
Are the timestamps parsed consistently?
```

## Why this matters

The business problem says:

```text
Using one month of data...
```

So time is not just another column. It becomes part of the ML key and later affects:

- anomaly detection
- failure prediction
- forecasting
- monitoring-cycle alignment

## Script Used

[time_range.py](C:/Users/navad/ML_data/EDA/time_range.py)

## Important Finding: Date Format Is Not Fully Consistent

Some timestamps look like:

```text
2/6/2026
```

This can mean:

```text
2 June 2026
```

or:

```text
6 February 2026
```

So the script tests both date interpretations:

- `day/month/year`
- `month/day/year`

Then it selects the one that best matches a one-month export.

## Results

| Dataset | Selected Format | Start Time | End Time | Unique Dates |
|---|---|---|---|---:|
| dell_idrac_health_export_20260703_mockup.csv | month/day/year | 2026-06-09 02:45 | 2026-07-09 22:47 | 31 |
| dell_idrac_health_ext_export_20260702_mockup.csv | day/month/year | 2026-06-02 02:47 | 2026-07-02 22:50 | 31 |
| dell_idrac_health_ext_export_20260703_mockup.csv | day/month/year | 2026-02-07 02:47 | 2026-03-07 22:50 | 29 |
| hpe_ilo_health_export_20260702_mockup.csv | day/month/year | 2026-06-02 02:02 | 2026-07-02 22:49 | 31 |
| hpe_ilo_health_export_20260703_mockup.csv | day/month/year | 2026-02-07 02:46 | 2026-03-07 22:46 | 29 |
| ping_status_export_20260702_mockup.csv | day/month/year | 2026-06-02 02:00 | 2026-07-02 22:59 | 31 |
| ping_status_export_20260703_mockup.csv | month/day/year | 2026-06-03 02:00 | 2026-07-03 22:59 | 31 |

## Interpretation

Most files contain approximately one month of data.

The cleanest aligned set for our next analyses is:

```text
ping_status_export_20260702_mockup.csv
hpe_ilo_health_export_20260702_mockup.csv
dell_idrac_health_ext_export_20260702_mockup.csv
```

These all cover:

```text
2026-06-02 to 2026-07-02
```

That makes them the best files to use while learning the merge logic.

## Current Conclusion

We should not blindly parse all files using one fixed date format.

For the step-by-step EDA, we should continue with the aligned `20260702` files first. Once the method is clear, we can extend it to all exports.

## Next Step

The next investigation should be:

```text
Step 7: Monitoring Frequency
```

Questions:

```text
How many observations does each machine have per day?
Do we see 6 observations per day?
Are the expected monitoring times 02:00, 06:00, 10:00, 14:00, 18:00, and 22:00?
```
