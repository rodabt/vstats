# A/B Test Readout

Runs a two-sample A/B test (`experiment.abtest`) on control vs treatment
conversions and renders the result as a bar chart with 95% confidence-interval
error bars, percentage value labels, gridlines, and a plain-English verdict in the
subtitle.

Running the example regenerates `ab_test_readout.svg` in this directory.

**Run:** `v run examples/ab-test-readout/main.v`

**Modules used:** `vstats.chart`, `vstats.experiment`

**Python equivalent:** `statsmodels` proportions z-test plus a `matplotlib` bar
chart with `yerr` error bars.
