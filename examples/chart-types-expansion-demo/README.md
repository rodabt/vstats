# Chart Types Expansion Demo

Exercises the five chart-types-expansion features together: a **combo** chart (bar on
the primary y-axis, line on a secondary y-axis, shared categorical x-axis), a
**waterfall** revenue bridge, a **grouped (clustered) bar** regional comparison, and a
**dumbbell** before/after latency comparison.

Running the example regenerates the four `.svg` files in this directory.

**Run:** `v run examples/chart-types-expansion-demo/main.v`

**Modules used:** `vstats.chart`

**Python equivalent:** matplotlib/plotly combo charts (`twinx()` + bar/line), a
waterfall via manual cumulative bar offsets, `plt.bar` with grouped positions, and a
dumbbell/slope plot via paired scatter + line segments.
