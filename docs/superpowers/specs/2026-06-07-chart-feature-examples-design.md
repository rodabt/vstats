# Design: chart feature showcase examples

**Date:** 2026-06-07
**Status:** Approved (design phase)
**Scope:** Two new example directories that showcase the expanded `chart` module features by charting real vstats analysis output.

## Goal

Add two scenario examples under `examples/`, each driven by a real vstats analysis, together exercising all seven features added in the chart expansion: CI bands, area fills, value labels, subtitle, gridlines, per-series color, and error bars.

## Example A — `examples/ab-test-readout/`

An experiment readout. Construct two 0/1 conversion arrays (control ≈ 12%, treatment ≈ 18%) programmatically, run `experiment.ab_test`, and chart the result.

- File: `ab_test_readout.svg`.
- Chart: a **bar** chart of the two arm conversion rates (`control_mean`, `treatment_mean`), with:
  - **error bars** — per-arm 95% CI half-width computed as `1.96 * std / sqrt(n)` from `control_std`/`treatment_std` and `n_control`/`n_treatment`;
  - **value labels** — each rate shown above its bar (custom `labels` formatted as percentages);
  - a **color** override on the bar series;
  - **gridlines** (`Theme{ grid: true }`);
  - a **subtitle** carrying the verdict: relative lift, p-value, and significant yes/no.
- Features demonstrated: bar, error bars, value labels, per-series color, gridlines, subtitle.

## Example B — `examples/retention-bands/`

Cohort retention with uncertainty. Hardcode 4 cohorts' retention counts over 6 periods, run `growth.create_cohort_analysis`, and chart two views.

- File `retention_bands.svg`:
  - the average retention **line** (`avg_retention` over period index);
  - a **CI band** between the cross-cohort min and max retention at each period (computed from `retention_matrix`);
  - each cohort drawn as a faint **line with a per-series color**;
  - **gridlines** and a **subtitle**.
- File `retention_area.svg`:
  - average retention as an **area fill** to the zero baseline;
  - **gridlines** and a **subtitle**.
- Features demonstrated: line, CI band, area fill, per-series color, gridlines, subtitle.

**Coverage:** CI bands (B), area fills (B), value labels (A), subtitle (A, B), gridlines (A, B), per-series color (A override, B cohort lines), error bars (A).

## File layout & conventions

Each directory follows the existing example convention:

- `main.v` — starts with a `// Scenario:` / `// Demonstrates:` / `// Python equivalent:` comment block, `module main`, `import vstats.chart` plus the analysis module (`vstats.experiment` or `vstats.growth`); prints a short `println` narration; writes SVGs to `os.dir(@FILE)`.
- `README.md` — title, one-paragraph description, a **Run** line, **Modules used**, **Python equivalent** (matching the other examples' format).
- Committed reference `.svg` outputs beside `main.v`.

## Data flow

**A:** build `control []f64` and `treatment []f64` of 0/1 outcomes (200 each, with fixed conversion counts) → `res := experiment.ab_test(control, treatment)` → bar values `[res.control_mean, res.treatment_mean]`; error array `[1.96*res.control_std/sqrt(n_control), 1.96*res.treatment_std/sqrt(n_treatment)]`; labels `['12.0%', '18.0%']`-style from the means; subtitle from `res.relative_lift`, `res.p_value`, `res.significant`.

**B:** `names`, `sizes`, `retention_data [][]int` (4 cohorts × 6 periods) → `ca := growth.create_cohort_analysis(names, sizes, retention_data)`. `xs` = period indices `0..5`. Average line = `ca.avg_retention`. Band: `lo[j] = min over cohorts of ca.retention_matrix[i][j]`, `hi[j] = max`. Cohort lines: each `ca.retention_matrix[i]`. Area chart uses `ca.avg_retention`.

## Error handling

`fn main()` with `!`-propagation on each `.save(path)!`. (`ab_test` and `create_cohort_analysis` return values, not results.)

## Verification

Both examples are fast and file-producing. The plan will:

- `v run examples/ab-test-readout/main.v` and `v run examples/retention-bands/main.v` exit 0 and print their narration;
- each `.svg` exists, starts with `<svg`, ends with `</svg>`;
- `ab_test_readout.svg` contains `<rect` bars plus extra `<line>` whiskers and percentage value-label text;
- `retention_bands.svg` contains `<polygon` (band) and multiple `<polyline>` (avg + cohort lines);
- `retention_area.svg` contains `<polygon` (area).

## Docs wiring

Add two sections to `docs/src/examples.md` (each with an `<!-- include: examples/<dir>/main.v -->` tag), change the intro count from "Eight" to "Ten", and rebuild HTML via `make docs`.

## Out of scope

- No changes to the `chart` module itself — examples use only the existing API.
- No raster output.
