# Cohort Retention with Uncertainty

Builds a cohort analysis (`growth.create_cohort_analysis`) from four monthly
cohorts and visualizes retention two ways: the average retention line with a
cross-cohort min/max **band** and faint per-cohort lines, and the average curve as
an **area** fill. Both use gridlines and a descriptive subtitle.

Running the example regenerates `retention_bands.svg` and `retention_area.svg`.

**Run:** `v run examples/retention-bands/main.v`

**Modules used:** `vstats.chart`, `vstats.growth`

**Python equivalent:** `matplotlib` `fill_between` for the retention band plus
per-cohort line plots.
