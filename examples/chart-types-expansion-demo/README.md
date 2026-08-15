# Chart Types Expansion Demo

Exercises the chart-types-expansion features together: a **combo** chart (bar on the
primary y-axis, line on a secondary y-axis, shared categorical x-axis, legend below the
plot), a **waterfall** revenue bridge (value labels always above each segment), a
**grouped (clustered) bar** regional comparison (value labels above each bar), a
**dumbbell** before/after latency comparison (x-axis forced to start at 100 via
`.xmin()`), and a **100% stacked bar** product-mix chart (each segment's percentage
centered inside it, in black or white text chosen by that segment's background
brightness, omitted entirely when the segment is too short to fit).

Running the example regenerates the five `.svg` files in this directory.

**Run:** `v run examples/chart-types-expansion-demo/main.v`

**Modules used:** `vstats.chart`

**Python equivalent:** matplotlib/plotly combo charts (`twinx()` + bar/line), a
waterfall via manual cumulative bar offsets, `plt.bar` with grouped positions, and a
dumbbell/slope plot via paired scatter + line segments.
