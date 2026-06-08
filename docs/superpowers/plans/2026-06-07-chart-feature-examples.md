# Chart Feature Showcase Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two example directories — `examples/ab-test-readout/` and `examples/retention-bands/` — that chart real vstats analysis output and together exercise all seven new chart features (CI bands, area fills, value labels, subtitle, gridlines, per-series color, error bars).

**Architecture:** Each directory is a self-contained `main.v` script plus `README.md` and committed reference SVGs, following the existing example convention. Example A drives a bar chart from `experiment.abtest`; example B drives a band/area chart from `growth.create_cohort_analysis`. A final task wires both into the docs site.

**Tech Stack:** V 0.5.1; modules `vstats.chart`, `vstats.experiment`, `vstats.growth`; stdlib `os`, `math`. Examples are verified by compiling and running, not unit-tested.

---

## Key facts the engineer must know

- **Example convention:** a directory under `examples/` with `main.v` (3-line `// Scenario:` / `// Demonstrates:` / `// Python equivalent:` header, then `module main`) and a `README.md`. Imports use the full module path: `import vstats.chart`, `import vstats.experiment`, `import vstats.growth`.
- **A/B API:** `experiment.abtest(control []f64, treatment []f64, cfg ABTestConfig) ABTestResult`. `ABTestConfig` is `@[params]` so the third argument may be omitted. `ABTestResult` fields used: `control_mean`, `treatment_mean`, `control_std`, `treatment_std`, `n_control`, `n_treatment`, `relative_lift`, `p_value`, `significant`.
- **Cohort API:** `growth.create_cohort_analysis(cohort_names []string, initial_sizes []int, retention_data [][]int) CohortAnalysis`. Fields used: `avg_retention []f64`, `retention_matrix [][]f64`.
- **chart API for these examples:** `chart.new(title:, subtitle:, width:, height:, theme:)` then chain `.bar(values, color:, show_values:, labels:, err:)`, `.line(x, y, label:, color:)`, `.band(x, lower, upper, label:)`, `.area(x, y, label:)`, `.xlabel(s)`, `.ylabel(s)`, `.save(path) !`. Turn on gridlines with `theme: chart.Theme{ grid: true }`.
- **Output location:** write SVGs to `os.dir(@FILE)` so they land beside `main.v`.
- **Running:** `v run examples/<dir>/main.v` (fast). Compile-check only: `v -check examples/<dir>/main.v`.
- The whole SVG is one line; in shell checks use `grep -o ... | wc -l`, not `grep -c`.
- `README.md` is exempt from the `*.md` gitignore via the `!README.md` rule, so `git add` works normally.

---

## Task 1: A/B readout example program

**Files:**
- Create: `examples/ab-test-readout/main.v`

- [ ] **Step 1: Write the program**

```v
// Scenario: A/B Test Readout
// Demonstrates: vstats.chart + vstats.experiment — bar chart with error bars, value labels, theming
// Python equivalent: statsmodels proportions z-test + matplotlib bar with yerr
module main

import os
import math
import vstats.experiment
import vstats.chart

fn main() {
	println('=== A/B Test Readout ===\n')

	// 0/1 conversion outcomes: control 24/200 = 12%, treatment 36/200 = 18%
	mut control := []f64{len: 200, init: 0.0}
	for i in 0 .. 24 {
		control[i] = 1.0
	}
	mut treatment := []f64{len: 200, init: 0.0}
	for i in 0 .. 36 {
		treatment[i] = 1.0
	}

	res := experiment.abtest(control, treatment)
	ci_c := 1.96 * res.control_std / math.sqrt(f64(res.n_control))
	ci_t := 1.96 * res.treatment_std / math.sqrt(f64(res.n_treatment))

	println('Control:   ${res.control_mean * 100:.1f}%  (n=${res.n_control})')
	println('Treatment: ${res.treatment_mean * 100:.1f}%  (n=${res.n_treatment})')
	println('Lift: ${res.relative_lift * 100:.1f}%   p=${res.p_value:.4f}   significant=${res.significant}')

	verdict := if res.significant {
		'Treatment lifts conversion ${res.relative_lift * 100:.1f}% (p=${res.p_value:.3f}, significant)'
	} else {
		'No significant difference (p=${res.p_value:.3f})'
	}

	out_dir := os.dir(@FILE)
	rates := [res.control_mean, res.treatment_mean]
	labels := ['${res.control_mean * 100:.1f}%', '${res.treatment_mean * 100:.1f}%']

	chart.new(title: 'A/B Test: Conversion Rate', subtitle: verdict, width: 560, height: 380,
		theme: chart.Theme{ grid: true })
		.bar(rates, color: '#2ca02c', show_values: true, labels: labels, err: [ci_c, ci_t])
		.xlabel('Arm (0 = control, 1 = treatment)')
		.ylabel('Conversion rate')
		.save(os.join_path(out_dir, 'ab_test_readout.svg'))!

	println('\nwrote 1 chart to ${out_dir}')
}
```

- [ ] **Step 2: Verify it compiles**

Run: `v -check examples/ab-test-readout/main.v`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add examples/ab-test-readout/main.v
git commit -m "feat(examples): add A/B test readout chart example"
```

---

## Task 2: Run the A/B example and commit its SVG

**Files:**
- Create (generated): `examples/ab-test-readout/ab_test_readout.svg`

- [ ] **Step 1: Run the example**

Run: `v run examples/ab-test-readout/main.v`
Expected stdout (numbers may vary slightly):
```
=== A/B Test Readout ===

Control:   12.0%  (n=200)
Treatment: 18.0%  (n=200)
Lift: ...%   p=...   significant=...

wrote 1 chart to examples/ab-test-readout
```

- [ ] **Step 2: Verify the SVG is well-formed and shows the features**

Run:
```bash
p=examples/ab-test-readout/ab_test_readout.svg
head -c4 "$p"; echo " <- start"; tail -c7 "$p"; echo " <- end"
echo "bars (rect): $(grep -o '<rect' "$p" | wc -l)"          # bg + 2 bars => >= 3
echo "lines: $(grep -o '<line' "$p" | wc -l)"                 # axes + ticks + 6 whisker lines
echo "value labels: $(grep -o '>12.0%<' "$p" | wc -l) $(grep -o '>18.0%<' "$p" | wc -l)"  # 1 and 1
echo "bar color: $(grep -o 'fill=\"#2ca02c\"' "$p" | head -1)"
echo "grid color: $(grep -o 'stroke=\"#e0e0e0\"' "$p" | head -1)"
```
Expected: starts `<svg`, ends `</svg>`; rect count ≥ 3; line count clearly more than just axes (whiskers present); both value labels found once; the bar color `#2ca02c` and grid color `#e0e0e0` are present.

- [ ] **Step 3: Commit the reference SVG**

```bash
git add examples/ab-test-readout/ab_test_readout.svg
git commit -m "feat(examples): add A/B readout reference SVG"
```

---

## Task 3: A/B readout README

**Files:**
- Create: `examples/ab-test-readout/README.md`

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add examples/ab-test-readout/README.md
git commit -m "docs(examples): add A/B readout README"
```

---

## Task 4: Retention bands example program

**Files:**
- Create: `examples/retention-bands/main.v`

- [ ] **Step 1: Write the program**

```v
// Scenario: Cohort Retention with Uncertainty
// Demonstrates: vstats.chart + vstats.growth — line + CI band, area fill, per-series color
// Python equivalent: matplotlib fill_between for retention bands + per-cohort lines
module main

import os
import vstats.growth
import vstats.chart

fn main() {
	println('=== Cohort Retention with Uncertainty ===\n')

	names := ['Jan', 'Feb', 'Mar', 'Apr']
	sizes := [1000, 1200, 900, 1100]
	// retained counts per period (period 0 = signup month)
	retention_data := [
		[1000, 720, 560, 470, 410, 380],
		[1200, 900, 740, 620, 560, 510],
		[900, 590, 430, 350, 300, 270],
		[1100, 800, 650, 560, 500, 460],
	]
	ca := growth.create_cohort_analysis(names, sizes, retention_data)

	periods := ca.avg_retention.len
	mut xs := []f64{len: periods}
	for j in 0 .. periods {
		xs[j] = f64(j)
	}

	// cross-cohort spread band (min..max retention at each period)
	mut lo := []f64{len: periods}
	mut hi := []f64{len: periods}
	for j in 0 .. periods {
		mut mn := ca.retention_matrix[0][j]
		mut mx := ca.retention_matrix[0][j]
		for i in 0 .. ca.retention_matrix.len {
			v := ca.retention_matrix[i][j]
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		lo[j] = mn
		hi[j] = mx
	}

	out_dir := os.dir(@FILE)

	// Chart 1: average retention line + cross-cohort band + faint per-cohort lines
	mut c := chart.new(title: 'Cohort Retention', subtitle: 'Average with cross-cohort min/max band',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
	c = c.band(xs, lo, hi, label: 'min/max')
	for i in 0 .. ca.retention_matrix.len {
		c = c.line(xs, ca.retention_matrix[i], color: '#cccccc')
	}
	c = c.line(xs, ca.avg_retention, label: 'average', color: '#1f77b4')
	c.xlabel('Months since signup')
		.ylabel('Retention')
		.save(os.join_path(out_dir, 'retention_bands.svg'))!

	// Chart 2: average retention as an area fill
	chart.new(title: 'Average Retention', subtitle: 'Area under the average retention curve',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.area(xs, ca.avg_retention, label: 'avg')
		.line(xs, ca.avg_retention, color: '#1f77b4')
		.xlabel('Months since signup')
		.ylabel('Retention')
		.save(os.join_path(out_dir, 'retention_area.svg'))!

	println('Final average retention: ${ca.avg_retention[periods - 1] * 100:.1f}%')
	println('wrote 2 charts to ${out_dir}')
}
```

- [ ] **Step 2: Verify it compiles**

Run: `v -check examples/retention-bands/main.v`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add examples/retention-bands/main.v
git commit -m "feat(examples): add cohort retention bands chart example"
```

---

## Task 5: Run the retention example and commit its SVGs

**Files:**
- Create (generated): `examples/retention-bands/retention_bands.svg`, `examples/retention-bands/retention_area.svg`

- [ ] **Step 1: Run the example**

Run: `v run examples/retention-bands/main.v`
Expected stdout:
```
=== Cohort Retention with Uncertainty ===

Final average retention: ...%
wrote 2 charts to examples/retention-bands
```

- [ ] **Step 2: Verify both SVGs are well-formed and show the features**

Run:
```bash
for f in retention_bands retention_area; do
  p="examples/retention-bands/$f.svg"
  head -c4 "$p"; echo " <- start $f"; tail -c7 "$p"; echo " <- end $f"
done
echo "band polygon: $(grep -o '<polygon' examples/retention-bands/retention_bands.svg | wc -l)"     # >= 1
echo "lines (avg + 4 cohorts): $(grep -o '<polyline' examples/retention-bands/retention_bands.svg | wc -l)"  # 5
echo "avg color: $(grep -o 'stroke=\"#1f77b4\"' examples/retention-bands/retention_bands.svg | head -1)"
echo "cohort color: $(grep -o 'stroke=\"#cccccc\"' examples/retention-bands/retention_bands.svg | head -1)"
echo "area polygon: $(grep -o '<polygon' examples/retention-bands/retention_area.svg | wc -l)"      # >= 1
```
Expected: both files start `<svg` / end `</svg>`; `retention_bands.svg` has ≥ 1 polygon (band) and 5 polylines (avg + 4 cohorts), with both `#1f77b4` and `#cccccc` present; `retention_area.svg` has ≥ 1 polygon (area).

- [ ] **Step 3: Commit the reference SVGs**

```bash
git add examples/retention-bands/retention_bands.svg examples/retention-bands/retention_area.svg
git commit -m "feat(examples): add retention bands reference SVGs"
```

---

## Task 6: Retention bands README

**Files:**
- Create: `examples/retention-bands/README.md`

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add examples/retention-bands/README.md
git commit -m "docs(examples): add retention bands README"
```

---

## Task 7: Wire both examples into the docs

**Files:**
- Modify: `docs/src/examples.md`

- [ ] **Step 1: Update the intro count**

In `docs/src/examples.md`, change the first word of the intro paragraph from `Eight` to `Ten`.

Find:
```markdown
Eight end-to-end scenarios, each targeting a different module and showcasing
```
Replace with:
```markdown
Ten end-to-end scenarios, each targeting a different module and showcasing
```

- [ ] **Step 2: Append the two new sections**

Add the following at the end of `docs/src/examples.md` (after the `chart-gallery` section):

```markdown

---

## ab-test-readout

A two-sample A/B test rendered as a bar chart with 95% CI error bars, percentage
value labels, gridlines, and a plain-English verdict subtitle.

<!-- include: examples/ab-test-readout/main.v -->

---

## retention-bands

Cohort retention with uncertainty: the average retention line with a cross-cohort
min/max band and faint per-cohort lines, plus the average curve as an area fill.

<!-- include: examples/retention-bands/main.v -->
```

- [ ] **Step 3: Verify the edits**

Run:
```bash
sed -n '3p' docs/src/examples.md | grep -q '^Ten' && echo "intro OK"
grep -q '<!-- include: examples/ab-test-readout/main.v -->' docs/src/examples.md && echo "ab include OK"
grep -q '<!-- include: examples/retention-bands/main.v -->' docs/src/examples.md && echo "retention include OK"
```
Expected: prints `intro OK`, `ab include OK`, `retention include OK`.

- [ ] **Step 4: Rebuild the HTML docs**

Run: `make docs`
Expected: `Done. 15 pages built.`

- [ ] **Step 5: Commit**

```bash
git add docs/src/examples.md docs/examples.html
git commit -m "docs: wire A/B readout and retention bands examples into examples.md"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Example A `ab-test-readout/` (bar + error bars + value labels + per-series color + gridlines + subtitle) → Tasks 1–3.
- Example B `retention-bands/` (line + CI band + area + per-series color + gridlines + subtitle, two SVGs) → Tasks 4–6.
- Real analysis data (`experiment.abtest`, `growth.create_cohort_analysis`) → Tasks 1, 4.
- Conventions: header comment, `import vstats.*`, `os.dir(@FILE)`, narration, `README.md`, committed SVGs → all example tasks.
- Error handling via `!`-propagation in `main` → Tasks 1, 4.
- Verification by running + checking SVG feature elements → Tasks 2, 5.
- Docs wiring (two include sections + count bump to "Ten" + rebuild) → Task 7.
- Feature coverage: CI bands (Task 4), area fills (Task 4), value labels (Task 1), subtitle (Tasks 1, 4), gridlines (Tasks 1, 4), per-series color (Tasks 1, 4), error bars (Task 1). All seven covered.

**Placeholder scan:** none — all code and commands are concrete.

**Type/name consistency:** uses the verified `experiment.abtest` and `growth.create_cohort_analysis` signatures and the real `ABTestResult` / `CohortAnalysis` field names; SVG filenames are identical across the program, verification, and README in each example.

**Risk note:** SVG element-count checks use `grep -o | wc -l` (the SVG is a single line); polyline count for `retention_bands.svg` is exactly 5 (4 cohort lines + 1 average) since the band renders as a `<polygon>`, not a polyline.
