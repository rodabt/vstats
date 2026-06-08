# Chart Gallery Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `examples/chart-gallery/` — a runnable example that fits regressions on the Boston Housing dataset and renders four SVG charts (scatter+line fit, residuals-vs-fitted, residuals histogram, coefficient bar) using the `chart` module.

**Architecture:** A single `main.v` script: load one dataset, fit a simple single-feature regression and a full multivariate regression, derive residuals, and emit four `.svg` files beside the source via the fluent `chart` API. A `README.md` and a `docs/src/examples.md` entry follow the repo's existing example conventions.

**Tech Stack:** V 0.5.1, modules `vstats.utils` (datasets), `vstats.ml` (regression), `vstats.chart` (SVG charts), stdlib `os` and `math`. Examples are verified by compiling/running (not unit-tested).

---

## Key facts the engineer must know

- **Example layout convention:** each example is a directory under `examples/` with `main.v` (starts with a 3-line `// Scenario:` / `// Demonstrates:` / `// Python equivalent:` comment block, then `module main`) and a `README.md`.
- **Import paths in examples** use the full module path: `import vstats.chart`, `import vstats.ml`, `import vstats.utils` (the in-repo `tests/` use bare names, but `examples/` use `vstats.`).
- **Boston Housing dataset** (`utils.load_boston_housing() !RegressionDataset`) is simplified to **3 features**: `feature_names == ["Crime Rate", "% Residential Land", "Distance to Employment"]`; `target` is `Median House Price`. Fields: `.features [][]f64`, `.target []f64`, `.feature_names []string`, `.name string`.
- **Regression API:** `ml.linear_regression[T](x [][]T, y []T) LinearModel[T]` where `LinearModel` has `.coefficients []T` and `.intercept T`; `ml.linear_predict[T](model, x [][]T) []T`; `ml.rmse[T](y_true, y_pred) f64`.
- **chart API:** `chart.new(title:, width:, height:, theme:)` then chain `.scatter(x, y, label:)`, `.line(x, y, label:)`, `.bar(values, label:)`, `.histogram(data, nbins:)`, `.xlabel(s)`, `.ylabel(s)`, `.axhline(y)`, `.save(path) !`. `chart.Theme{...}` overrides styling; `background` and `palette` are rendered, but **`grid` is not implemented** — do not rely on it.
- **Output location:** write SVGs to the example's own directory using `os.dir(@FILE)` so they land beside `main.v`.
- **Running:** `v run examples/chart-gallery/main.v` (fast); compile-check only: `v -check examples/chart-gallery/main.v`.

---

## Task 1: Write the example program (main.v)

**Files:**
- Create: `examples/chart-gallery/main.v`

- [ ] **Step 1: Write the complete example**

```v
// Scenario: Chart Gallery — Regression Diagnostics
// Demonstrates: vstats.chart — scatter, line, bar, histogram, legend, guide line, theming
// Python equivalent: matplotlib regression diagnostics (scatter+fit, residuals, hist, coef bar)
module main

import os
import math
import vstats.utils
import vstats.ml
import vstats.chart

fn main() {
	println('=== Chart Gallery: Regression Diagnostics ===\n')

	// --- Setup: one dataset, two regressions ---
	dataset := utils.load_boston_housing()!
	x := dataset.features // [][]f64, 3 features
	y := dataset.target   // []f64, median house price
	names := dataset.feature_names
	println('Dataset: ${dataset.name} (${y.len} samples, ${names.len} features)')

	out_dir := os.dir(@FILE) // directory of this source file

	// Single-feature regression: price ~ Crime Rate (feature index 0)
	crime := x.map(it[0])
	x1 := crime.map([it]) // [][]f64 single-feature design matrix
	model := ml.linear_regression(x1, y)
	pred := ml.linear_predict(model, x1)
	rmse := ml.rmse(y, pred)
	println('Simple model (price ~ Crime Rate): intercept=${model.intercept:.2f}, slope=${model.coefficients[0]:.3f}, RMSE=${rmse:.2f}')

	// residuals
	mut resid := []f64{len: y.len}
	for i in 0 .. y.len {
		resid[i] = y[i] - pred[i]
	}

	// --- Chart 1: scatter (observed) + line (fit) ---
	// sort points by crime so the fitted line draws cleanly left-to-right
	mut order := []int{len: crime.len, init: index}
	order.sort(crime[a] < crime[b])
	xs := order.map(crime[it])
	ys := order.map(pred[it])
	chart.new(title: 'Price vs Crime Rate', width: 640, height: 420)
		.scatter(crime, y, label: 'observed')
		.line(xs, ys, label: 'fit')
		.xlabel('Crime Rate')
		.ylabel('Median House Price')
		.save(os.join_path(out_dir, 'regression_fit.svg'))!

	// --- Chart 2: residuals vs fitted, with a zero reference line ---
	chart.new(title: 'Residuals vs Fitted', width: 640, height: 420)
		.scatter(pred, resid, label: 'residual')
		.axhline(0.0)
		.xlabel('Fitted value')
		.ylabel('Residual')
		.save(os.join_path(out_dir, 'residuals_vs_fitted.svg'))!

	// --- Chart 3: histogram of residuals (auto bins) ---
	chart.new(title: 'Residual Distribution', width: 640, height: 420)
		.histogram(resid)
		.xlabel('Residual')
		.ylabel('Count')
		.save(os.join_path(out_dir, 'residuals_hist.svg'))!

	// --- Chart 4: coefficient bar from the full multivariate regression ---
	full := ml.linear_regression(x, y)
	coefs := full.coefficients
	mut top := 0
	for j in 1 .. coefs.len {
		if math.abs(coefs[j]) > math.abs(coefs[top]) {
			top = j
		}
	}
	println('Full model strongest predictor: ${names[top]} (coef=${coefs[top]:.3f})')

	custom := chart.Theme{
		background: '#f7f7f7'
		palette:    ['#756bb1']
	}
	chart.new(title: 'Regression Coefficients', width: 640, height: 420, theme: custom)
		.bar(coefs, label: 'coefficient')
		.xlabel('Feature index (0=Crime, 1=ResLand, 2=Distance)')
		.ylabel('Coefficient')
		.save(os.join_path(out_dir, 'coefficients.svg'))!

	println('\nwrote 4 charts to ${out_dir}')
}
```

- [ ] **Step 2: Verify it compiles**

Run: `v -check examples/chart-gallery/main.v`
Expected: no output, exit code 0 (compiles cleanly).

If it errors on `order.sort(crime[a] < crime[b])`, that is the V custom-sort closure form; confirm the file matches the code above exactly. If it errors on `crime.map([it])`, verify the brackets — `[it]` wraps each `f64` into a one-element `[]f64`.

- [ ] **Step 3: Commit**

```bash
git add examples/chart-gallery/main.v
git commit -m "feat(examples): add chart-gallery regression diagnostics example"
```

---

## Task 2: Run the example and commit the generated SVGs

**Files:**
- Create (generated): `examples/chart-gallery/regression_fit.svg`, `residuals_vs_fitted.svg`, `residuals_hist.svg`, `coefficients.svg`

- [ ] **Step 1: Run the example**

Run: `v run examples/chart-gallery/main.v`
Expected stdout (numbers will vary):

```
=== Chart Gallery: Regression Diagnostics ===

Dataset: Boston Housing (506 samples, 3 features)
Simple model (price ~ Crime Rate): intercept=..., slope=..., RMSE=...
Full model strongest predictor: ... (coef=...)

wrote 4 charts to examples/chart-gallery
```

- [ ] **Step 2: Verify all four SVGs exist and are well-formed**

Run:
```bash
for f in regression_fit residuals_vs_fitted residuals_hist coefficients; do
  p="examples/chart-gallery/$f.svg"
  head -c 4 "$p"; echo "  <- start of $f.svg"
  tail -c 7 "$p"; echo "  <- end of $f.svg"
done
```
Expected: each file starts with `<svg` and ends with `</svg>`.

- [ ] **Step 3: Spot-check chart contents**

Run:
```bash
grep -c '<polyline' examples/chart-gallery/regression_fit.svg   # expect >= 1 (the fit line)
grep -c '<circle'   examples/chart-gallery/regression_fit.svg   # expect 506 (observed points)
grep -o 'fill="#756bb1"' examples/chart-gallery/coefficients.svg | head -1  # custom palette applied
grep -o 'fill="#f7f7f7"' examples/chart-gallery/coefficients.svg | head -1  # custom background applied
```
Expected: polyline count ≥ 1, circle count = 506, and both custom color strings are found in `coefficients.svg`.

- [ ] **Step 4: Commit the reference SVGs**

```bash
git add examples/chart-gallery/regression_fit.svg examples/chart-gallery/residuals_vs_fitted.svg examples/chart-gallery/residuals_hist.svg examples/chart-gallery/coefficients.svg
git commit -m "feat(examples): add chart-gallery reference SVG outputs"
```

---

## Task 3: Write the README

**Files:**
- Create: `examples/chart-gallery/README.md`

- [ ] **Step 1: Write the README**

```markdown
# Chart Gallery — Regression Diagnostics

Fits a regression on the Boston Housing dataset and renders four SVG charts that
together cover every `chart` type: a scatter of observed prices vs crime rate with
the fitted **line** overlaid, a **scatter** of residuals vs fitted values with a
zero **guide line**, a **histogram** of residuals, and a **bar** chart of the
multivariate regression coefficients with a custom theme.

Running the example regenerates the four `.svg` files in this directory.

**Run:** `v run examples/chart-gallery/main.v`

**Modules used:** `vstats.chart`, `vstats.ml`, `vstats.utils`

**Python equivalent:** matplotlib regression-diagnostics plots — `scatter` + fitted
line, residuals-vs-fitted, `hist` of residuals, and a coefficient `bar` chart.
```

- [ ] **Step 2: Verify the README is tracked despite the `*.md` gitignore rule**

Run: `git check-ignore examples/chart-gallery/README.md; echo "exit=$?"`
Expected: `exit=1` (NOT ignored — the `!README.md` rule in `.gitignore` un-ignores it). If it prints the path with `exit=0`, force-add with `git add -f examples/chart-gallery/README.md`.

- [ ] **Step 3: Commit**

```bash
git add examples/chart-gallery/README.md
git commit -m "docs(examples): add chart-gallery README"
```

---

## Task 4: Wire the example into the docs

**Files:**
- Modify: `docs/src/examples.md`

- [ ] **Step 1: Update the intro count**

In `docs/src/examples.md`, change the first paragraph's opening word from `Seven` to `Eight`:

Find:
```markdown
Seven end-to-end scenarios, each targeting a different module and showcasing
```
Replace with:
```markdown
Eight end-to-end scenarios, each targeting a different module and showcasing
```

- [ ] **Step 2: Append the new scenario section**

Add the following at the end of `docs/src/examples.md` (after the `hypothesis-battery` section):

```markdown

---

## chart-gallery

Regression diagnostics rendered to SVG: observed-vs-fitted scatter with the fit
line, residuals-vs-fitted with a zero guide, a residual histogram, and a
coefficient bar chart with a custom theme — every `chart` type in one example.

<!-- include: examples/chart-gallery/main.v -->
```

- [ ] **Step 3: Verify the edits**

Run:
```bash
head -4 docs/src/examples.md | grep -q '^Eight' && echo "intro OK"
grep -q '<!-- include: examples/chart-gallery/main.v -->' docs/src/examples.md && echo "include OK"
```
Expected: prints `intro OK` and `include OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/src/examples.md
git commit -m "docs: wire chart-gallery example into examples.md"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- New `examples/chart-gallery/` dir with `main.v`, `README.md`, four committed SVGs → Tasks 1, 2, 3.
- Four charts (scatter+line fit, residuals-vs-fitted+axhline, residual histogram, coefficient bar+custom theme) → Task 1 (all four), verified in Task 2.
- All four chart types + legend (two labeled series in chart 1) + axhline + labels/title + theme override + save-to-disk → Task 1.
- Single dataset (Boston Housing), simple + full regression, residuals derived → Task 1 setup.
- stdout narration (size, RMSE, intercept, strongest coefficient, "wrote 4 charts") → Task 1, verified Task 2 Step 1.
- Error handling via `!`-propagation in `main` → Task 1 (`load_boston_housing()!`, each `.save(...)!`).
- Verification by running + checking SVGs → Task 2.
- Docs wiring with include tag + count bump → Task 4.
- Out-of-scope items (no unit tests, no new chart features, no raster) → respected; the plan adds no chart-module code and the grid no-op is avoided by using palette/background.

**Placeholder scan:** none — all code and commands are concrete.

**Type/name consistency:** field/method names (`features`, `target`, `feature_names`, `name`, `coefficients`, `intercept`, `linear_regression`, `linear_predict`, `rmse`, `chart.new`, `.scatter/.line/.bar/.histogram/.xlabel/.ylabel/.axhline/.save`, `chart.Theme{background, palette}`) match the verified module APIs and are used identically across tasks. SVG filenames are identical in Tasks 1, 2, and the README description.

**Known risks flagged inline:** V custom-sort and `map([it])` forms (Task 1 Step 2); `*.md` gitignore vs `!README.md` (Task 3 Step 2).
