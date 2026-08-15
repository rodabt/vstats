# Chart Types Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add categorical x-axis labels, a secondary y-axis (combo charts), waterfall,
grouped (clustered) bar, and dumbbell (before/after) chart types to the `chart` module.

**Architecture:** All five features extend the existing `Chart` → `Series` →
`Geom`/`Scene` → SVG pipeline in `chart/chart.v`, `chart/scales.v`, `chart/scene.v`,
`chart/svg.v`, `chart/theme.v`. New chart types are new `SeriesKind` variants with
builder methods following the existing pattern (`chart.line(...)`, `.bar(...)`, etc);
the categorical x-axis and secondary y-axis are chart-level/opt-level additions that
existing series kinds also participate in.

**Tech Stack:** V language, no external dependencies. Tests live in `tests/` and assert
on rendered SVG strings (`.contains(...)`, `.count(...)`), matching the existing style in
`tests/chart_test.v`.

## Global Constraints

- No dependencies added — `chart` module stays dependency-free (only `math`, `os`,
  `strconv`, `strings` from V's stdlib, per existing imports).
- Naming: `snake_case` functions/variables, `PascalCase` structs/enums (per
  `CLAUDE.md` conventions).
- Tests live in `tests/`, not alongside source files; prefix `test_` /
  `fn test__...()` matching existing chart test file style.
- Existing chart output must remain byte-identical for charts that don't use any new
  feature (no `xcategories()` call, no `secondary_axis: true`, no new series kinds) —
  every task that touches shared code (`draw_ticks`, `data_bounds`, `effective_margins`)
  must preserve this via the existing test suite passing unchanged.
- Backward compatibility: the existing per-series `labels` fallback for x-axis
  categorical detection (`stacked_bar`, `heatmap`) stays in place; `xcategories()` is
  additive, not a replacement requiring migration.
- Run `v test tests/` after every task; all pre-existing tests plus new ones must pass
  before moving to the next task.

---

## File Structure

| File | Change |
|------|--------|
| `chart/scene.v` | No changes (existing `Meta`, `Line`, `Rect`, `Circle`, `Polyline`, `Text` primitives are sufficient for all 5 features). |
| `chart/theme.v` | Add 3 waterfall color fields to `Theme`. |
| `chart/chart.v` | Add `SeriesKind.waterfall/.grouped_bar/.dumbbell`; `Series.secondary_axis` field; `Chart.xcategories_` field + `.xcategories()` method; `SeriesOpts` new fields; 3 new builder methods (`waterfall`, `grouped_bar`, `dumbbell`); `series_bounds()` cases for the 3 new kinds; `data_bounds()` split into primary/secondary; `Geom` gains `yscale2`/`has_secondary`; `draw_series()` cases for the 3 new kinds + secondary-axis scale selection; `draw_ticks()` categorical-x path generalized + secondary-axis right-side ticks; `effective_margins()` grows `margin_right` for secondary axis; 3 new `*_meta()` helpers; `draw_value_labels()` cases for the 3 new kinds. |
| `tests/chart_xcategories_test.v` | New: categorical x-axis tests. |
| `tests/chart_dualaxis_test.v` | New: secondary y-axis tests. |
| `tests/chart_waterfall_test.v` | New: waterfall tests. |
| `tests/chart_grouped_bar_test.v` | New: grouped bar tests. |
| `tests/chart_dumbbell_test.v` | New: dumbbell tests. |

Task order: shared infrastructure first (categorical x-axis, then secondary axis), since
grouped_bar/waterfall/dumbbell reuse the categorical-x-axis code path, and the combo use
case (bar + line on secondary axis) is easiest to verify once both are in place. Then the
three new chart kinds, each independently testable.

---

## Task 1: Categorical x-axis (`Chart.xcategories`)

**Files:**
- Modify: `chart/chart.v` (`Chart` struct ~line 40-52, `draw_ticks` ~line 1260-1380)
- Test: `tests/chart_xcategories_test.v`

**Interfaces:**
- Consumes: existing `Chart` struct, `Series.kind`/`Series.labels`, `g.xscale`,
  `nice_ticks`, `fmt_tick` (all already defined in `chart/chart.v` / `chart/scales.v`).
- Produces: `Chart.xcategories_ []string` field; `pub fn (c Chart) xcategories(labels []string) Chart`.
  Later tasks (grouped_bar, waterfall) rely on `xcategories_` being checked in
  `draw_ticks` ahead of the existing `x_cat_labels` per-series-kind scan.

- [ ] **Step 1: Write the failing test**

Create `tests/chart_xcategories_test.v`:

```v
import chart

fn test__xcategories_renders_labels_on_x_axis() {
	svg := chart.new(width: 400, height: 300)
		.xcategories(['Jan', 'Feb', 'Mar'])
		.line([0.0, 1.0, 2.0], [10.0, 20.0, 15.0])
		.render()
	assert svg.contains('>Jan<')
	assert svg.contains('>Feb<')
	assert svg.contains('>Mar<')
	// numeric ticks should not appear for the x-axis when categories are set
	assert !svg.contains('>0.5<')
}

fn test__xcategories_works_with_bar() {
	svg := chart.new(width: 400, height: 300)
		.xcategories(['A', 'B'])
		.bar([5.0, 9.0])
		.render()
	assert svg.contains('>A<')
	assert svg.contains('>B<')
}

fn test__no_xcategories_keeps_numeric_ticks() {
	svg := chart.new(width: 400, height: 300)
		.line([0.0, 1.0, 2.0], [0.0, 50.0, 100.0])
		.render()
	assert svg.contains('>100<') // numeric tick, unaffected by this feature
}

fn test__xcategories_does_not_break_existing_stacked_bar_labels() {
	// stacked_bar's own per-series labels fallback still works when xcategories_ unset
	svg := chart.new(width: 400, height: 300)
		.stacked_bar([[1.0, 2.0], [3.0, 4.0]], labels: ['G1', 'G2'])
		.render()
	assert svg.contains('>G1<')
	assert svg.contains('>G2<')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_xcategories_test.v`
Expected: FAIL — `xcategories` is not a method on `chart.Chart` (compile error).

- [ ] **Step 3: Implement `xcategories_` field and method**

In `chart/chart.v`, add the field to the `Chart` struct (after `vlines []f64` around
line 51):

```v
pub struct Chart {
mut:
	title        string
	subtitle     string
	width        int
	height       int
	theme        Theme
	xlabel_      string
	ylabel_      string
	series       []Series
	hlines       []f64
	vlines       []f64
	xcategories_ []string
}
```

Add the builder method, near `xlabel`/`ylabel` (~line 202-212):

```v
pub fn (c Chart) xcategories(labels []string) Chart {
	mut nc := c
	nc.xcategories_ = labels.clone()
	return nc
}
```

- [ ] **Step 4: Generalize the categorical x-axis path in `draw_ticks`**

In `chart/chart.v`, `draw_ticks` (~line 1260-1330), replace the x-axis branch. Currently:

```v
	// x-axis ticks
	if x_cat_labels.len > 0 {
		for i, lbl in x_cat_labels {
```

Change the condition to prefer `c.xcategories_` when set, falling back to the existing
per-series scan otherwise:

```v
	// x-axis ticks
	x_labels_to_use := if c.xcategories_.len > 0 { c.xcategories_ } else { x_cat_labels }
	if x_labels_to_use.len > 0 {
		for i, lbl in x_labels_to_use {
```

(the rest of that loop body — drawing the tick `Line` and `Text` — is unchanged, just
swap `x_cat_labels` for `x_labels_to_use` inside it). Leave the `y_cat_labels` /
y-axis branch and the `x_cat_labels`-gathering loop above it untouched — this task only
adds an override, it does not remove the existing per-series detection (needed for
backward compatibility per Global Constraints).

- [ ] **Step 5: Run tests to verify they pass**

Run: `v test tests/chart_xcategories_test.v`
Expected: PASS (all 4 tests)

Run: `v test tests/` (full suite)
Expected: PASS — no regressions in existing chart tests.

- [ ] **Step 6: Commit**

```bash
git add chart/chart.v tests/chart_xcategories_test.v
git commit -m "feat(chart): add Chart.xcategories() for shared categorical x-axis"
```

---

## Task 2: Secondary y-axis (combo charts)

**Files:**
- Modify: `chart/chart.v` (`Series` struct, `SeriesOpts` struct, `Geom` struct, all
  series-constructor methods, `data_bounds`, `geometry`, `draw_series`,
  `draw_error_bars`, `draw_value_labels`, `draw_ticks`, `effective_margins`)
- Test: `tests/chart_dualaxis_test.v`

**Interfaces:**
- Consumes: `Series` struct (Task-agnostic, existing), `Geom`/`LinearScale` (from
  `chart/scales.v`), `series_bounds()` (existing, per-kind bounds function).
- Produces: `Series.secondary_axis bool`; `SeriesOpts.secondary_axis bool`; `Geom.yscale2
  LinearScale`; `Geom.has_secondary bool`; `Chart.data_bounds_secondary() (f64, f64, f64, f64)`.
  Later tasks (waterfall/grouped_bar/dumbbell builder methods) must set
  `secondary_axis: opts.secondary_axis` when constructing their `Series` literal, exactly
  like every other builder method added in this task.

- [ ] **Step 1: Write the failing test**

Create `tests/chart_dualaxis_test.v`:

```v
import chart

fn test__secondary_axis_draws_right_side_ticks() {
	svg := chart.new(width: 500, height: 300)
		.bar([100.0, 200.0, 150.0])
		.line([0.0, 1.0, 2.0], [0.1, 0.5, 0.3], secondary_axis: true)
		.render()
	// right-anchored ticks only appear when a secondary axis is drawn
	assert svg.contains('text-anchor="start"')
	// secondary domain (0.1..0.5) tick should render, distinct from primary (100..200) domain
	assert svg.contains('>0.5<') || svg.contains('>0.50<')
}

fn test__no_secondary_axis_renders_unchanged() {
	svg := chart.new(width: 400, height: 300)
		.line([0.0, 1.0], [0.0, 1.0])
		.render()
	assert !svg.contains('text-anchor="start"') // no right-axis labels, no legend, no title
}

fn test__secondary_axis_series_uses_independent_scale() {
	// primary series spans 0..1000, secondary spans 0..1 -- if they shared one scale,
	// the secondary line would be squashed to a flat near-zero line. Assert both
	// axes' extreme tick values render distinctly.
	svg := chart.new(width: 500, height: 300)
		.bar([0.0, 1000.0])
		.line([0.0, 1.0], [0.0, 1.0], secondary_axis: true)
		.render()
	assert svg.contains('>1000<')
	assert svg.contains('>1<')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_dualaxis_test.v`
Expected: FAIL — `secondary_axis` is not a recognized field on `SeriesOpts` (compile error).

- [ ] **Step 3: Add `secondary_axis` to `Series` and `SeriesOpts`**

In `chart/chart.v`, add to the `Series` struct (~line 23-38):

```v
struct Series {
	kind           SeriesKind
	x              []f64
	y              []f64
	lo             []f64
	hi             []f64
	err            []f64
	label          string
	color          string
	color_lo       string
	color_hi       string
	nbins          int
	show_values    bool
	labels         []string
	colors         []string
	secondary_axis bool
}
```

Add to `SeriesOpts` (~line 64-73):

```v
@[params]
pub struct SeriesOpts {
pub:
	label          string
	color          string
	show_values    bool
	labels         []string
	err            []f64
	colors         []string
	secondary_axis bool
}
```

Then, in every existing series-constructor method that builds a `Series{...}` literal
(`line`, `scatter`, `bar`, `area`, `step` — the ones plausible for combo use; `histogram`,
`band`, `box`, `dot`, `violin`, `hbar`, `heatmap`, `stacked_bar` also get the field for
consistency), add `secondary_axis: opts.secondary_axis` to the literal. Concretely, in
`chart/chart.v`:

- `line` (~line 95): add `secondary_axis: opts.secondary_axis,` after `labels: opts.labels.clone(),`
- `scatter` (~line 119): same
- `bar` (~line 160): same
- `area` (~line 251): same
- `step` (~line 266): same

(Other kinds like `histogram`/`dot`/`hbar`/`heatmap`/`stacked_bar`/`box`/`violin`/`band`
don't need it for this task's tests, but for consistency add it to all of them —
`SeriesOpts.secondary_axis` defaults to `false` so this is a no-op for callers that don't
set it. Do this for every method that takes `SeriesOpts` as its `opts` parameter.)

- [ ] **Step 4: Split `data_bounds` into primary/secondary**

In `chart/chart.v`, rename the existing `data_bounds` body to operate on a filtered
series list, and add a secondary variant. Replace (~line 714-747):

```v
fn (c Chart) data_bounds() (f64, f64, f64, f64) {
	mut xmin := 0.0
	mut xmax := 1.0
	mut ymin := 0.0
	mut ymax := 1.0
	mut first := true
	for s in c.series {
		bx0, bx1, by0, by1 := series_bounds(s)
		if first {
			xmin, xmax, ymin, ymax = bx0, bx1, by0, by1
			first = false
		} else {
			if bx0 < xmin {
				xmin = bx0
			}
			if bx1 > xmax {
				xmax = bx1
			}
			if by0 < ymin {
				ymin = by0
			}
			if by1 > ymax {
				ymax = by1
			}
		}
	}
	if xmax == xmin {
		xmax = xmin + 1.0
	}
	if ymax == ymin {
		ymax = ymin + 1.0
	}
	return xmin, xmax, ymin, ymax
}
```

with:

```v
fn bounds_for(series []Series) (f64, f64, f64, f64) {
	mut xmin := 0.0
	mut xmax := 1.0
	mut ymin := 0.0
	mut ymax := 1.0
	mut first := true
	for s in series {
		bx0, bx1, by0, by1 := series_bounds(s)
		if first {
			xmin, xmax, ymin, ymax = bx0, bx1, by0, by1
			first = false
		} else {
			if bx0 < xmin {
				xmin = bx0
			}
			if bx1 > xmax {
				xmax = bx1
			}
			if by0 < ymin {
				ymin = by0
			}
			if by1 > ymax {
				ymax = by1
			}
		}
	}
	if xmax == xmin {
		xmax = xmin + 1.0
	}
	if ymax == ymin {
		ymax = ymin + 1.0
	}
	return xmin, xmax, ymin, ymax
}

fn (c Chart) data_bounds() (f64, f64, f64, f64) {
	mut primary := []Series{}
	for s in c.series {
		if !s.secondary_axis {
			primary << s
		}
	}
	if primary.len == 0 {
		primary = c.series.clone()
	}
	return bounds_for(primary)
}

fn (c Chart) data_bounds_secondary() (f64, f64, f64, f64) {
	mut secondary := []Series{}
	for s in c.series {
		if s.secondary_axis {
			secondary << s
		}
	}
	return bounds_for(secondary)
}

fn (c Chart) has_secondary_series() bool {
	for s in c.series {
		if s.secondary_axis {
			return true
		}
	}
	return false
}
```

`primary.len == 0 { primary = c.series.clone() }` guards the edge case where every series
in a chart is (incorrectly) marked secondary — `data_bounds()` still returns a sane
domain instead of the `0..1` default, matching prior behavior when nothing is excluded.

- [ ] **Step 5: Add `yscale2`/`has_secondary` to `Geom` and compute them in `geometry()`**

In `chart/chart.v`, add fields to `Geom` (~line 426-437):

```v
struct Geom {
	plot_x        f64
	plot_y        f64
	plot_w        f64
	plot_h        f64
	xmin          f64
	xmax          f64
	ymin          f64
	ymax          f64
	xscale        LinearScale
	yscale        LinearScale
	yscale2       LinearScale
	has_secondary bool
}
```

Update `geometry()` (~line 809-838):

```v
fn (c Chart) geometry() Geom {
	ml, mr, mt, mb := c.effective_margins()
	plot_x := f64(ml)
	plot_y := f64(mt)
	plot_w := f64(c.width - ml - mr)
	plot_h := f64(c.height - mt - mb)
	xmin, xmax, ymin, ymax := c.data_bounds()
	has_secondary := c.has_secondary_series()
	mut yscale2 := LinearScale{}
	if has_secondary {
		_, _, symin, symax := c.data_bounds_secondary()
		yscale2 = LinearScale{
			domain_min: symin
			domain_max: symax
			range_min:  plot_y + plot_h
			range_max:  plot_y
		}
	}
	return Geom{
		plot_x:        plot_x
		plot_y:        plot_y
		plot_w:        plot_w
		plot_h:        plot_h
		xmin:          xmin
		xmax:          xmax
		ymin:          ymin
		ymax:          ymax
		xscale:        LinearScale{
			domain_min: xmin
			domain_max: xmax
			range_min:  plot_x
			range_max:  plot_x + plot_w
		}
		yscale:        LinearScale{
			domain_min: ymin
			domain_max: ymax
			range_min:  plot_y + plot_h
			range_max:  plot_y
		}
		yscale2:       yscale2
		has_secondary: has_secondary
	}
}
```

- [ ] **Step 6: Use `yscale2` for secondary-axis series in `draw_series`, `draw_error_bars`, `draw_value_labels`**

In `chart/chart.v`, `draw_series` (~line 898-1237), for the `.line`, `.scatter`, `.step`,
`.bar`, `.area` cases (the kinds combo charts realistically mix), replace the flat
`g.yscale.map(...)` calls with a per-series scale pick. Add this helper right above
`draw_series`:

```v
fn (g Geom) yscale_for(s Series) LinearScale {
	if s.secondary_axis && g.has_secondary {
		return g.yscale2
	}
	return g.yscale
}
```

Then in each relevant match arm, before using `g.yscale`, bind `ys := g.yscale_for(s)`
and replace `g.yscale.map(...)` with `ys.map(...)` for that series' points. Example for
`.line` (~line 988-1001):

```v
			.line {
				ys := g.yscale_for(s)
				mut pts := []Point{}
				for i in 0 .. s.x.len {
					pts << Point{
						x: g.xscale.map(s.x[i])
						y: ys.map(s.y[i])
					}
				}
				scene.primitives << Polyline{
					points: pts
					stroke: s.color
					width:  t.series_width
				}
			}
```

Apply the same `ys := g.yscale_for(s)` + swap pattern to `.scatter`, `.step`, `.bar`,
`.area` (their baseline/point mappings that currently call `g.yscale.map(...)`).

Do the same in `draw_error_bars` (~line 1466-1509): bind `ys := g.yscale_for(s)` inside
the `for s in c.series` loop, before computing `y_hi`/`y_lo`, and use `ys.map(...)`.

And in `draw_value_labels` (~line 1511-1652): for the `.line, .scatter`, `.bar`, `.step`
match arms, bind `ys := g.yscale_for(s)` and use `ys.map(...)` in place of `g.yscale.map(...)`.

- [ ] **Step 7: Draw right-side axis ticks when `has_secondary`**

In `chart/chart.v`, `draw_ticks` (~line 1260-1380), after the existing y-axis tick block
(the `if y_cat_labels.len > 0 { ... } else { ... }` for numeric y ticks), add:

```v
	// secondary y-axis ticks (right side)
	if g.has_secondary {
		for tk in nice_ticks(g.yscale2.domain_min, g.yscale2.domain_max, 5) {
			if tk < g.yscale2.domain_min - 1.0e-9 || tk > g.yscale2.domain_max + 1.0e-9 {
				continue
			}
			py := g.yscale2.map(tk)
			scene.primitives << Line{
				x1:     g.plot_x + g.plot_w
				y1:     py
				x2:     g.plot_x + g.plot_w + 5.0
				y2:     py
				stroke: t.axis_color
				width:  t.axis_width
			}
			scene.primitives << Text{
				x:       g.plot_x + g.plot_w + 8.0
				y:       py + 4.0
				content: fmt_tick(tk)
				size:    t.font_size
				fill:    t.axis_color
				anchor:  .start
				family:  t.font_family
			}
		}
	}
```

- [ ] **Step 8: Grow `margin_right` for the secondary axis in `effective_margins`**

In `chart/chart.v`, `effective_margins()` (~line 751-807), inside the "right" section
(after the existing loop computing `max_right` from value labels and legend), add before
`need_right := int(max_right) + 8`:

```v
	if c.has_secondary_series() {
		_, _, symin, symax := c.data_bounds_secondary()
		for tk in nice_ticks(symin, symax, 5) {
			w, _ := text_extent(fmt_tick(tk), t.font_size)
			// ticks sit further right than value labels: gap (13px) + tick text
			total := w + 13.0
			if total > max_right {
				max_right = total
			}
		}
	}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `v test tests/chart_dualaxis_test.v`
Expected: PASS (all 3 tests)

Run: `v test tests/` (full suite)
Expected: PASS — no regressions (charts with zero secondary-axis series compute
`has_secondary_series() == false`, skip all new code paths, produce identical output).

- [ ] **Step 10: Commit**

```bash
git add chart/chart.v tests/chart_dualaxis_test.v
git commit -m "feat(chart): add secondary y-axis for combo/dual-axis charts"
```

---

## Task 3: Waterfall chart

**Files:**
- Modify: `chart/theme.v` (3 new color fields), `chart/chart.v` (`SeriesKind`,
  `SeriesOpts`, builder method, `series_bounds`, `draw_series`, `draw_value_labels`, new
  meta helper)
- Test: `tests/chart_waterfall_test.v`

**Interfaces:**
- Consumes: `Series`/`SeriesOpts` (Task 2), `Theme` (this task extends it),
  `Chart.xcategories_`-compatible index positioning (Task 1's `draw_ticks` path works
  automatically once bars are positioned at integer indices — no direct dependency,
  waterfall bars are always index-positioned the same way `bar` already is).
- Produces: `pub fn (c Chart) waterfall(values []f64, labels []string, opts SeriesOpts) Chart`.

- [ ] **Step 1: Write the failing test**

Create `tests/chart_waterfall_test.v`:

```v
import chart

fn test__waterfall_emits_bars_and_labels() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([100.0, 20.0, -30.0, 90.0], ['Start', 'Gain', 'Loss', 'End'])
		.render()
	assert svg.count('<rect') >= 4 // background + 4 bars
	assert svg.contains('>Start<')
	assert svg.contains('>End<')
}

fn test__waterfall_uses_semantic_colors() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([100.0, 20.0, -30.0, 90.0], ['Start', 'Gain', 'Loss', 'End'])
		.render()
	assert svg.contains('#2ca02c') // increase (default)
	assert svg.contains('#d62728') // decrease (default)
	assert svg.contains('#7f7f7f') // total (default)
}

fn test__waterfall_color_overrides() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([10.0, 5.0, 15.0], ['Start', 'Gain', 'End'], color_increase: '#00ff00', color_total: '#000000')
		.render()
	assert svg.contains('#00ff00')
	assert svg.contains('#000000')
}

fn test__waterfall_show_values_labels_deltas() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([100.0, 20.0, -30.0, 90.0], ['Start', 'Gain', 'Loss', 'End'], show_values: true)
		.render()
	assert svg.contains('>+20<')
	assert svg.contains('>-30<')
}

fn test__waterfall_two_point_totals_only() {
	svg := chart.new(width: 400, height: 300)
		.waterfall([50.0, 80.0], ['Start', 'End'])
		.render()
	assert svg.count('<rect') >= 2
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_waterfall_test.v`
Expected: FAIL — `waterfall` is not a method on `chart.Chart` (compile error).

- [ ] **Step 3: Add semantic colors to `Theme`**

In `chart/theme.v`, add to the `Theme` struct:

```v
	waterfall_increase string = '#2ca02c'
	waterfall_decrease string = '#d62728'
	waterfall_total    string = '#7f7f7f'
```

(insert after `palette` field, before the closing `}`.)

- [ ] **Step 4: Add `.waterfall` to `SeriesKind` and color overrides to `SeriesOpts`**

In `chart/chart.v`, add to the `SeriesKind` enum (~line 7-21):

```v
enum SeriesKind {
	line
	scatter
	bar
	histogram
	band
	area
	step
	box_plot
	dot
	violin
	hbar
	heatmap
	stacked_bar
	waterfall
	grouped_bar
	dumbbell
}
```

(this task only uses `.waterfall`; `.grouped_bar` and `.dumbbell` are added here too so
the enum is defined once — Tasks 4 and 5 will implement their behavior but the compiler
requires all `match` arms over `SeriesKind` to be exhaustive, so every `match s.kind`
block touched in this task must also grow a `.grouped_bar` / `.dumbbell` arm; add trivial
`else {}`-safe stubs now, filled in by Tasks 4/5 — see Step 6 note below.)

Add to `SeriesOpts` (~line 64-73):

```v
	color_increase string
	color_decrease string
	color_total    string
```

- [ ] **Step 5: Add the `waterfall` builder method**

In `chart/chart.v`, after `stacked_bar` (~line 397-422):

```v
struct WaterfallMeta {
	is_total bool
	running  f64
}

pub fn (c Chart) waterfall(values []f64, labels []string, opts SeriesOpts) Chart {
	assert values.len >= 2
	assert labels.len == values.len
	mut nc := c
	mut sv := c.series.clone()
	col_inc := if opts.color_increase != '' { opts.color_increase } else { c.theme.waterfall_increase }
	col_dec := if opts.color_decrease != '' { opts.color_decrease } else { c.theme.waterfall_decrease }
	col_tot := if opts.color_total != '' { opts.color_total } else { c.theme.waterfall_total }
	sv << Series{
		kind:           .waterfall
		y:              values.clone()
		labels:         labels.clone()
		color:          col_inc // default series color slot; per-bar color resolved at draw time
		color_lo:       col_dec
		color_hi:       col_tot
		show_values:    opts.show_values
		label:          opts.label
		secondary_axis: opts.secondary_axis
	}
	nc.series = sv
	return nc
}
```

(reusing `Series.color`/`color_lo`/`color_hi` to carry the 3 semantic colors through to
draw time, following the same "reuse existing fields" pattern the spec calls out for
`.band`/`.dumbbell`.)

- [ ] **Step 6: Add `.waterfall` (and stub `.grouped_bar`/`.dumbbell`) to `series_bounds`**

In `chart/chart.v`, `series_bounds` (~line 607-712) is a `match s.kind { ... }`
expression — every arm must be covered. Add:

```v
		.waterfall {
			mut cum := 0.0
			mut ymin := 0.0
			mut ymax := 0.0
			for i, v in s.y {
				if i == 0 || i == s.y.len - 1 {
					cum = v
				} else {
					cum += v
				}
				if cum < ymin {
					ymin = cum
				}
				if cum > ymax {
					ymax = cum
				}
			}
			-0.5, f64(s.y.len) - 0.5, ymin, ymax
		}
		.grouped_bar {
			-0.5, 0.5, 0.0, 1.0 // implemented in Task 4
		}
		.dumbbell {
			-0.5, 0.5, 0.0, 1.0 // implemented in Task 5
		}
```

- [ ] **Step 7: Draw waterfall bars, connectors, and colors in `draw_series`**

In `chart/chart.v`, `draw_series`'s pass-2 `match s.kind { ... }` (~line 986-1237) has no
catch-all arm, so every `SeriesKind` variant needs an explicit arm for the module to
compile. Add the following 3 arms before the closing `.stacked_bar { ... }` arm — only
`.waterfall` has real logic in this task, `.grouped_bar`/`.dumbbell` are empty stubs
replaced with real logic in Tasks 4/5:

```v
			.grouped_bar {}
			.dumbbell {}
			.waterfall {
				ys := g.yscale_for(s)
				band := g.xscale.map(1.0) - g.xscale.map(0.0)
				bw := band * 0.8
				mut cum := 0.0
				mut prev_top_px := 0.0
				for i, v in s.y {
					is_total := i == 0 || i == s.y.len - 1
					mut base := cum
					if is_total {
						cum = v
						base = 0.0
					} else {
						cum += v
					}
					top := ys.map(math.max(base, cum))
					bottom := ys.map(math.min(base, cum))
					col := if is_total {
						s.color_hi
					} else if v >= 0.0 {
						s.color
					} else {
						s.color_lo
					}
					cx := g.xscale.map(f64(i))
					lbl := if s.labels.len > i { s.labels[i] } else { fmt_tick(f64(i)) }
					scene.primitives << Rect{
						x:      cx - bw / 2.0
						y:      top
						w:      bw
						h:      bottom - top
						fill:   col
						stroke: 'none'
						width:  0.0
						meta:   waterfall_meta(s.label, lbl, v, is_total, cum)
					}
					if i > 0 {
						scene.primitives << Line{
							x1:     g.xscale.map(f64(i - 1)) + bw / 2.0
							y1:     prev_top_px
							x2:     cx - bw / 2.0
							y2:     top
							stroke: t.grid_color
							width:  t.axis_width
						}
					}
					prev_top_px = top
				}
			}
```

Add the tooltip helper near the other `*_meta()` functions (~line 552-605):

```v
fn waterfall_meta(series string, label string, delta f64, is_total bool, running f64) Meta {
	head := if series != '' { '${series}\n' } else { '' }
	body := if is_total {
		'${label}: ${fmt_tick(running)}'
	} else {
		sign := if delta >= 0.0 { '+' } else { '' }
		'${label}: ${sign}${fmt_tick(delta)} (running: ${fmt_tick(running)})'
	}
	return Meta{
		tooltip: '${head}${body}'
		series:  series
		label:   label
		y:       fmt_tick(running)
	}
}
```

- [ ] **Step 8: Add value labels in `draw_value_labels`**

In `chart/chart.v`, `draw_value_labels`'s `match s.kind { ... }` (~line 1517-1650), add
before `else {}`:

```v
			.waterfall {
				ys := g.yscale_for(s)
				mut cum := 0.0
				for i, v in s.y {
					is_total := i == 0 || i == s.y.len - 1
					if is_total {
						cum = v
					} else {
						cum += v
					}
					text := if is_total {
						fmt_tick(cum)
					} else if v >= 0.0 {
						'+${fmt_tick(v)}'
					} else {
						fmt_tick(v)
					}
					scene.primitives << Text{
						x:       g.xscale.map(f64(i))
						y:       ys.map(cum) - 4.0
						content: text
						size:    t.font_size
						fill:    t.axis_color
						anchor:  .middle
						family:  t.font_family
					}
				}
			}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `v test tests/chart_waterfall_test.v`
Expected: PASS (all 5 tests)

Run: `v test tests/`
Expected: PASS — no regressions.

- [ ] **Step 10: Commit**

```bash
git add chart/chart.v chart/theme.v tests/chart_waterfall_test.v
git commit -m "feat(chart): add waterfall chart type"
```

---

## Task 4: Grouped (clustered) bar

**Files:**
- Modify: `chart/chart.v` (builder method, `series_bounds`, `draw_series`,
  `draw_ticks` categorical-x fallback list, `draw_value_labels`)
- Test: `tests/chart_grouped_bar_test.v`

**Interfaces:**
- Consumes: `SeriesKind.grouped_bar` (added in Task 3, stub bounds/draw arms already
  present), `Series.x`/`Series.nbins`/`Series.colors`/`Series.labels` (existing fields,
  same shape `stacked_bar` uses).
- Produces: `pub fn (c Chart) grouped_bar(groups [][]f64, opts SeriesOpts) Chart`.

- [ ] **Step 1: Write the failing test**

Create `tests/chart_grouped_bar_test.v`:

```v
import chart

fn test__grouped_bar_emits_one_rect_per_segment() {
	svg := chart.new(width: 500, height: 300)
		.grouped_bar([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]], labels: ['A', 'B', 'C'])
		.render()
	assert svg.count('<rect') >= 7 // background + 3 groups * 2 segments
	assert svg.contains('>A<')
	assert svg.contains('>B<')
	assert svg.contains('>C<')
}

fn test__grouped_bar_uses_per_segment_colors() {
	svg := chart.new(width: 500, height: 300)
		.grouped_bar([[1.0, 2.0]], colors: ['#111111', '#222222'])
		.render()
	assert svg.contains('#111111')
	assert svg.contains('#222222')
}

fn test__grouped_bar_y_bound_is_max_segment_not_stack_sum() {
	// stacked_bar with [10, 10] would need a y-axis reaching 20; grouped_bar only
	// needs to reach 10 since bars sit side by side, not stacked.
	svg := chart.new(width: 500, height: 300)
		.grouped_bar([[10.0, 10.0]])
		.render()
	assert !svg.contains('>20<')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_grouped_bar_test.v`
Expected: FAIL — `grouped_bar` is not a method on `chart.Chart` (compile error).

- [ ] **Step 3: Add the `grouped_bar` builder method**

In `chart/chart.v`, after `stacked_bar` (or after `waterfall` from Task 3, either
location is fine — keep new chart-type methods grouped together):

```v
pub fn (c Chart) grouped_bar(groups [][]f64, opts SeriesOpts) Chart {
	assert groups.len > 0
	nseg := groups[0].len
	assert nseg > 0
	mut flat := []f64{cap: groups.len * nseg}
	for grp in groups {
		assert grp.len == nseg
		for v in grp {
			flat << v
		}
	}
	mut nc := c
	mut sv := c.series.clone()
	sv << Series{
		kind:           .grouped_bar
		x:              flat
		nbins:          nseg
		label:          opts.label
		color:          ''
		labels:         opts.labels.clone()
		colors:         opts.colors.clone()
		show_values:    opts.show_values
		secondary_axis: opts.secondary_axis
	}
	nc.series = sv
	return nc
}
```

- [ ] **Step 4: Replace the `.grouped_bar` stub in `series_bounds`**

In `chart/chart.v`, `series_bounds`, replace the Task 3 stub:

```v
		.grouped_bar {
			-0.5, 0.5, 0.0, 1.0 // implemented in Task 4
		}
```

with:

```v
		.grouped_bar {
			nseg := s.nbins
			if nseg == 0 {
				return -0.5, 0.5, 0.0, 1.0
			}
			nbars := s.x.len / nseg
			mut max_seg := 0.0
			for v in s.x {
				if v > max_seg {
					max_seg = v
				}
			}
			-0.5, f64(nbars) - 0.5, 0.0, max_seg
		}
```

- [ ] **Step 5: Replace the `.grouped_bar` stub in `draw_series`**

In `chart/chart.v`, `draw_series`, replace the Task 3 stub arm with:

```v
			.grouped_bar {
				ys := g.yscale_for(s)
				nseg := s.nbins
				if nseg == 0 {
					continue
				}
				nbars := s.x.len / nseg
				band := g.xscale.map(1.0) - g.xscale.map(0.0)
				bw := band * 0.8
				sub_bw := bw / f64(nseg)
				baseline := ys.map(0.0)
				for i in 0 .. nbars {
					cx := g.xscale.map(f64(i))
					bar_lbl := if s.labels.len > i { s.labels[i] } else { fmt_tick(f64(i)) }
					for j in 0 .. nseg {
						seg_val := s.x[i * nseg + j]
						top := ys.map(seg_val)
						col := if s.colors.len > j { s.colors[j] } else { c.theme.color(j) }
						scene.primitives << Rect{
							x:      cx - bw / 2.0 + f64(j) * sub_bw
							y:      math.min(top, baseline)
							w:      sub_bw
							h:      math.abs(baseline - top)
							fill:   col
							stroke: 'none'
							width:  0.0
							meta:   seg_meta(bar_lbl, j, seg_val)
						}
					}
				}
			}
```

(reuses `seg_meta` already defined for `stacked_bar`.)

- [ ] **Step 6: Add `.grouped_bar` to the categorical-x-label detection fallback**

In `chart/chart.v`, `draw_ticks`, the `x_cat_labels`-gathering loop (~line 1271-1273):

```v
		if s.kind == .stacked_bar && s.labels.len > 0 && x_cat_labels.len == 0 {
			x_cat_labels = s.labels.clone()
		}
```

Change to:

```v
		if s.kind in [.stacked_bar, .grouped_bar] && s.labels.len > 0 && x_cat_labels.len == 0 {
			x_cat_labels = s.labels.clone()
		}
```

- [ ] **Step 7: Add `.grouped_bar` value labels in `draw_value_labels`**

In `chart/chart.v`, `draw_value_labels`, replace the Task 3 stub-equivalent (there was
no stub added for `.grouped_bar` value labels since `else {}` already covers it) by
adding a real arm before `else {}`:

```v
			.grouped_bar {
				ys := g.yscale_for(s)
				nseg := s.nbins
				if nseg == 0 {
					continue
				}
				nbars := s.x.len / nseg
				band := g.xscale.map(1.0) - g.xscale.map(0.0)
				bw := band * 0.8
				sub_bw := bw / f64(nseg)
				for i in 0 .. nbars {
					cx := g.xscale.map(f64(i))
					for j in 0 .. nseg {
						seg_val := s.x[i * nseg + j]
						top := ys.map(seg_val)
						scene.primitives << Text{
							x:       cx - bw / 2.0 + (f64(j) + 0.5) * sub_bw
							y:       top - 4.0
							content: fmt_tick(seg_val)
							size:    t.font_size
							fill:    t.axis_color
							anchor:  .middle
							family:  t.font_family
						}
					}
				}
			}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `v test tests/chart_grouped_bar_test.v`
Expected: PASS (all 3 tests)

Run: `v test tests/`
Expected: PASS — no regressions.

- [ ] **Step 9: Commit**

```bash
git add chart/chart.v tests/chart_grouped_bar_test.v
git commit -m "feat(chart): add grouped (clustered) bar chart type"
```

---

## Task 5: Dumbbell (before/after) plot

**Files:**
- Modify: `chart/chart.v` (`SeriesOpts.color_hi` note, builder method, `series_bounds`,
  `draw_series`, `draw_ticks` y-axis categorical detection, `draw_value_labels`)
- Test: `tests/chart_dumbbell_test.v`

**Interfaces:**
- Consumes: `SeriesKind.dumbbell` (added in Task 3, stub bounds/draw arms already
  present), `Series.lo`/`Series.hi`/`Series.labels` (existing fields, reused per the
  `.band` pattern).
- Produces: `pub fn (c Chart) dumbbell(before []f64, after []f64, labels []string, opts SeriesOpts) Chart`.

- [ ] **Step 1: Write the failing test**

Create `tests/chart_dumbbell_test.v`:

```v
import chart

fn test__dumbbell_emits_two_circles_and_connector_per_row() {
	svg := chart.new(width: 500, height: 300)
		.dumbbell([10.0, 20.0, 15.0], [25.0, 22.0, 30.0], ['Q1', 'Q2', 'Q3'])
		.render()
	assert svg.count('<circle') == 6 // 3 rows * 2 dots
	assert svg.contains('>Q1<')
	assert svg.contains('>Q2<')
	assert svg.contains('>Q3<')
}

fn test__dumbbell_before_after_distinct_colors() {
	svg := chart.new(width: 500, height: 300)
		.dumbbell([10.0], [20.0], ['Only'], color: '#111111', color_hi: '#222222')
		.render()
	assert svg.contains('#111111')
	assert svg.contains('#222222')
}

fn test__dumbbell_show_values_labels_both_points() {
	svg := chart.new(width: 500, height: 300)
		.dumbbell([10.0], [25.0], ['Row'], show_values: true)
		.render()
	assert svg.contains('>10<')
	assert svg.contains('>25<')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_dumbbell_test.v`
Expected: FAIL — `dumbbell` is not a method on `chart.Chart`, and `color_hi` is not a
recognized field on `SeriesOpts` (compile errors).

- [ ] **Step 3: Add `color_hi` to `SeriesOpts`**

In `chart/chart.v`, add to `SeriesOpts` (alongside the Task 3 `color_increase` etc.
fields):

```v
	color_hi string
```

- [ ] **Step 4: Add the `dumbbell` builder method**

In `chart/chart.v`, after `grouped_bar`:

```v
pub fn (c Chart) dumbbell(before []f64, after []f64, labels []string, opts SeriesOpts) Chart {
	assert before.len == after.len
	assert labels.len == before.len
	mut nc := c
	mut sv := c.series.clone()
	before_col := if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
	after_col := if opts.color_hi != '' { opts.color_hi } else { c.theme.color(c.series.len + 1) }
	sv << Series{
		kind:           .dumbbell
		lo:             before.clone()
		hi:             after.clone()
		labels:         labels.clone()
		label:          opts.label
		color:          before_col
		color_hi:       after_col
		show_values:    opts.show_values
		secondary_axis: opts.secondary_axis
	}
	nc.series = sv
	return nc
}
```

- [ ] **Step 5: Replace the `.dumbbell` stub in `series_bounds`**

In `chart/chart.v`, `series_bounds`, replace the Task 3 stub:

```v
		.dumbbell {
			-0.5, 0.5, 0.0, 1.0 // implemented in Task 5
		}
```

with:

```v
		.dumbbell {
			lo_before, hi_before := extent(s.lo)
			lo_after, hi_after := extent(s.hi)
			xlo := math.min(lo_before, lo_after)
			xhi := math.max(hi_before, hi_after)
			xlo, xhi, -0.5, f64(s.lo.len) - 0.5
		}
```

- [ ] **Step 6: Replace the `.dumbbell` stub in `draw_series`**

In `chart/chart.v`, `draw_series`, replace the Task 3 stub arm with:

```v
			.dumbbell {
				ys := g.yscale_for(s)
				for i in 0 .. s.lo.len {
					py := ys.map(f64(s.lo.len - 1 - i))
					before_px := g.xscale.map(s.lo[i])
					after_px := g.xscale.map(s.hi[i])
					lbl := if s.labels.len > i { s.labels[i] } else { fmt_tick(f64(i)) }
					scene.primitives << Line{
						x1:     before_px
						y1:     py
						x2:     after_px
						y2:     py
						stroke: t.grid_color
						width:  t.axis_width
					}
					scene.primitives << Circle{
						cx:     before_px
						cy:     py
						r:      t.marker_radius + 1.0
						fill:   s.color
						stroke: 'none'
						width:  0.0
						meta:   dumbbell_meta(s.label, lbl, s.lo[i], s.hi[i])
					}
					scene.primitives << Circle{
						cx:     after_px
						cy:     py
						r:      t.marker_radius + 1.0
						fill:   s.color_hi
						stroke: 'none'
						width:  0.0
						meta:   dumbbell_meta(s.label, lbl, s.lo[i], s.hi[i])
					}
				}
			}
```

Add the tooltip helper near `waterfall_meta` / other `*_meta()` functions:

```v
fn dumbbell_meta(series string, label string, before f64, after f64) Meta {
	head := if series != '' { '${series}\n' } else { '' }
	delta := after - before
	sign := if delta >= 0.0 { '+' } else { '' }
	return Meta{
		tooltip: '${head}${label}: ${fmt_tick(before)} → ${fmt_tick(after)} (${sign}${fmt_tick(delta)})'
		series:  series
		label:   label
		y:       fmt_tick(after)
	}
}
```

- [ ] **Step 7: Add `.dumbbell` to the y-axis categorical label detection**

In `chart/chart.v`, `draw_ticks`, the `y_cat_labels`-gathering loop (~line 1267-1269):

```v
		if s.kind in [.dot, .hbar] && s.labels.len > 0 && y_cat_labels.len == 0 {
			y_cat_labels = s.labels.clone()
		}
```

Change to:

```v
		if s.kind in [.dot, .hbar, .dumbbell] && s.labels.len > 0 && y_cat_labels.len == 0 {
			y_cat_labels = s.labels.clone()
		}
```

Also, `effective_margins()` computes `max_left` from `.dot`/`.hbar` labels (~line
767-776):

```v
	for s in c.series {
		if s.kind in [.dot, .hbar] {
			for lb in s.labels {
				w, _ := text_extent(lb, t.font_size)
				if w > max_left {
					max_left = w
				}
			}
		}
	}
```

Change to:

```v
	for s in c.series {
		if s.kind in [.dot, .hbar, .dumbbell] {
			for lb in s.labels {
				w, _ := text_extent(lb, t.font_size)
				if w > max_left {
					max_left = w
				}
			}
		}
	}
```

- [ ] **Step 8: Add `.dumbbell` value labels in `draw_value_labels`**

In `chart/chart.v`, `draw_value_labels`, add before `else {}`:

```v
			.dumbbell {
				ys := g.yscale_for(s)
				for i in 0 .. s.lo.len {
					py := ys.map(f64(s.lo.len - 1 - i))
					before_px := g.xscale.map(s.lo[i])
					after_px := g.xscale.map(s.hi[i])
					scene.primitives << Text{
						x:       before_px
						y:       py - t.marker_radius - 4.0
						content: fmt_tick(s.lo[i])
						size:    t.font_size
						fill:    t.axis_color
						anchor:  .middle
						family:  t.font_family
					}
					scene.primitives << Text{
						x:       after_px
						y:       py - t.marker_radius - 4.0
						content: fmt_tick(s.hi[i])
						size:    t.font_size
						fill:    t.axis_color
						anchor:  .middle
						family:  t.font_family
					}
				}
			}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `v test tests/chart_dumbbell_test.v`
Expected: PASS (all 3 tests)

Run: `v test tests/` (full suite)
Expected: PASS — no regressions across the entire module.

- [ ] **Step 10: Commit**

```bash
git add chart/chart.v tests/chart_dumbbell_test.v
git commit -m "feat(chart): add dumbbell (before/after) chart type"
```

---

## Task 6: Combo-chart integration example

**Files:**
- Create: `examples/chart_combo_waterfall_demo.v`
- Test: none (this is a runnable example, verified by execution, matching the pattern
  of other `examples/*.v` files in the repo — no `tests/` entry needed)

**Interfaces:**
- Consumes: `chart.new()`, `.bar()`, `.line(..., secondary_axis: true)`,
  `.xcategories()`, `.waterfall()`, `.grouped_bar()`, `.dumbbell()` — all public APIs
  from Tasks 1-5.
- Produces: `.svg` files under `os.temp_dir()` demonstrating each new feature, confirming
  the whole module compiles and runs end-to-end together (not just in isolated unit
  tests).

- [ ] **Step 1: Check existing example file conventions**

Run: `find /home/rabt/devel/vstats/examples -iname "*chart*"` (or equivalent listing) to
confirm naming/output conventions before writing — mirror whatever pattern existing
chart examples use for `.save(path)` calls and `println` confirmations.

- [ ] **Step 2: Write the example**

Create `examples/chart_combo_waterfall_demo.v`:

```v
import chart
import os

fn main() {
	out_dir := os.temp_dir()

	// combo: bar (primary axis, revenue) + line (secondary axis, conversion rate)
	combo := chart.new(title: 'Revenue vs Conversion Rate', width: 500, height: 320)
		.xcategories(['Jan', 'Feb', 'Mar', 'Apr'])
		.bar([120.0, 150.0, 90.0, 200.0], label: 'Revenue')
		.line([0.0, 1.0, 2.0, 3.0], [0.12, 0.18, 0.09, 0.22], label: 'Conv. Rate', secondary_axis: true)
	combo.save(os.join_path(out_dir, 'combo_demo.svg')) or { panic(err) }

	// waterfall: revenue bridge
	wf := chart.new(title: 'Revenue Bridge', width: 500, height: 320)
		.waterfall([1000.0, 250.0, -80.0, 150.0, -40.0, 1280.0], [
			'Start',
			'New Sales',
			'Churn',
			'Upsell',
			'Refunds',
			'End',
		], show_values: true)
	wf.save(os.join_path(out_dir, 'waterfall_demo.svg')) or { panic(err) }

	// grouped bar: quarterly comparison across regions
	gb := chart.new(title: 'Sales by Region', width: 500, height: 320)
		.xcategories(['Q1', 'Q2', 'Q3', 'Q4'])
		.grouped_bar([[30.0, 45.0], [40.0, 38.0], [35.0, 50.0], [50.0, 42.0]], labels: [
			'Q1',
			'Q2',
			'Q3',
			'Q4',
		], colors: ['#1f77b4', '#ff7f0e'])
	gb.save(os.join_path(out_dir, 'grouped_bar_demo.svg')) or { panic(err) }

	// dumbbell: before/after intervention
	db := chart.new(title: 'Latency Before/After Optimization (ms)', width: 500, height: 320)
		.dumbbell([320.0, 410.0, 275.0, 500.0], [180.0, 260.0, 190.0, 300.0], [
			'API A',
			'API B',
			'API C',
			'API D',
		], show_values: true)
	db.save(os.join_path(out_dir, 'dumbbell_demo.svg')) or { panic(err) }

	println('Wrote 4 demo charts to ${out_dir}')
}
```

- [ ] **Step 3: Run it**

Run: `v run examples/chart_combo_waterfall_demo.v`
Expected: prints `Wrote 4 demo charts to ...` with no panics; 4 `.svg` files exist in
the temp dir afterward. Manually open at least one (e.g. `waterfall_demo.svg`) in a
browser to visually confirm bars/colors/connectors render sensibly.

- [ ] **Step 4: Run the full test suite one more time**

Run: `v test tests/`
Expected: PASS — final confirmation nothing regressed across the whole plan.

- [ ] **Step 5: Commit**

```bash
git add examples/chart_combo_waterfall_demo.v
git commit -m "docs(chart): add combo/waterfall/grouped-bar/dumbbell example"
```

---

## Self-Review Notes

- **Spec coverage:** All 5 spec parts have a task — Part 1 (categorical x-axis) → Task 1,
  Part 2 (secondary axis) → Task 2, Part 3 (waterfall) → Task 3, Part 4 (grouped bar) →
  Task 4, Part 5 (dumbbell) → Task 5. Task 6 exercises them together per the spec's
  combo-chart use case.
- **Enum exhaustiveness:** `SeriesKind` gains all 3 new variants in Task 3 (Step 4) since
  V requires exhaustive `match` over enums; every `match s.kind` block touched anywhere
  in the plan (`series_bounds`, `draw_series`, `draw_value_labels`) gets stub arms for
  `.grouped_bar`/`.dumbbell` immediately in Task 3, replaced with real logic in Tasks 4/5
  — this avoids compile breakage between tasks.
- **Type consistency:** `g.yscale_for(s)` (introduced Task 2 Step 6) is reused verbatim
  by Tasks 3/4/5's `draw_series`/`draw_value_labels` additions. `seg_meta` (existing,
  used by `stacked_bar`) is reused by Task 4 rather than duplicated. `extent()` (existing
  helper) is reused by Task 5's `series_bounds` arm.
- **Backward compatibility:** Task 1 keeps the old per-series `x_cat_labels` scan as a
  fallback; Task 2's `yscale_for` returns `g.yscale` unless `secondary_axis && has_secondary`,
  so untouched chart calls take the exact old path.
