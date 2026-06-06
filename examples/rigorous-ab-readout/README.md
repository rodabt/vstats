# Rigorous A/B Test Readout

End-to-end experiment analysis pipeline covering the four checks a rigorous
readout requires: SRM detection, outlier handling, variance reduction (CUPED),
and multiple testing correction (Benjamini-Hochberg).

**Run:** `v run examples/rigorous-ab-readout/main.v`

**Modules used:** `vstats.experiment`, `vstats.stats`

**Python equivalent:** `scipy.stats` + `statsmodels` + `statsmodels.stats.multitest`
+ custom SRM logic — four separate libraries, ~80 lines.
