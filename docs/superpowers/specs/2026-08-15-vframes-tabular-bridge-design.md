# vframes Tabular Bridge — Design

**Date:** 2026-08-15
**Module:** `utils/` (new file) + `examples/` (new example)
**Status:** Approved design, ready for implementation plan

## Goal

Establish a best-practice pattern for feeding data loaded via
[vframes](https://github.com/rodabt/vframes) (a DataFrame library for V, backed by
DuckDB — CSV/JSON/parquet/Excel/DB reads, `filter`/`group_by`/`join`) into vstats'
`ml`/`stats`/`hypothesis` functions, **without vstats taking on vframes as a
dependency**. vstats' `v.mod` declares `dependencies: []` and `CLAUDE.md` states the
library is dependency-free by design; this work must preserve that for the core
library and for `make test` / `v test tests/`.

## Background

Of the three candidate projects considered for integration (cuiqData, vframes, asql),
vframes was chosen: it fills a concrete, current gap (`utils/datasets.v` only has a
handful of hardcoded loaders; `ml`/`stats` functions take raw `[]f64`/`[][]f64`), it
carries the smallest dependency footprint of the three (vframes itself depends only on
`vduckdb`), and — critically for this design — the actual integration work can be done
as a **structural, generic bridge** on the vstats side rather than a hard import.

vframes' `DataFrame.values()!` returns `Data = []map[string]json2.Any | []map[string]string`
(row-oriented). vstats' `utils.Dataset` / `utils.RegressionDataset` structs (already
used by every loader in `utils/datasets.v`) hold `features [][]f64` plus either
`target []int` (classification) or `target []f64` (regression) — exactly the shape
`ml.linear_regression[T](x [][]T, y []T)` and friends consume.

## Design

### Part 1 — `utils/tabular.v` (new file, `utils` module, zero new imports)

Pure V functions operating on `[]map[string]f64` — a generic row-oriented shape any
DataFrame library (vframes or otherwise) can trivially produce, with no compile-time
knowledge of vframes:

```v
pub fn rows_to_vector(rows []map[string]f64, column string) ![]f64
pub fn rows_to_matrix(rows []map[string]f64, columns []string) ![][]f64
pub fn rows_to_dataset(rows []map[string]f64, feature_cols []string, target_col string, name string) !Dataset
pub fn rows_to_regression_dataset(rows []map[string]f64, feature_cols []string, target_col string, name string) !RegressionDataset
```

- **`rows_to_vector`**: returns an error if `rows` is empty, or if `column` is absent
  from any row (checked per-row, not just the first — a partially-populated column is
  as much a bug as a wholly-missing one and must not silently degrade to `0.0`).
- **`rows_to_matrix`**: builds `[][]f64` in **row-major** order —
  `result[i][j]` = row `i`, column `columns[j]` — matching how `ml` functions already
  consume feature matrices. Same missing-column error handling as `rows_to_vector`.
- **`rows_to_dataset`**: calls `rows_to_matrix` for `feature_cols` and
  `rows_to_vector` for `target_col`, then truncates each target value to `int` (V's
  `int(v)` cast) since `Dataset.target` is `[]int` (classification labels).
  `feature_names` is set to `feature_cols`, `target_name` to `target_col`, `name` and
  `description` from the `name` parameter (description left empty — caller-supplied
  data has no canned description to give).
- **`rows_to_regression_dataset`**: same shape, but the target stays `f64` — no cast.

These four functions are the entire surface. No `DataFrame`-like type, no I/O, no
vframes/json2 import anywhere in `utils/`.

### Part 2 — `tests/tabular_test.v` (new file)

Standard vstats test file — plain V, literal `[]map[string]f64` fixtures standing in
for what a real DataFrame's `.values()` would produce. Fits `make test` /
`v test tests/` with zero exceptions: no new dependency touches the default test run.

Cases: happy path for each of the 4 functions; missing-column error for
`rows_to_vector`/`rows_to_matrix`; empty-`rows` error; `rows_to_dataset`'s int
truncation (e.g. target `2.9` → `2`).

### Part 3 — `examples/vframes-integration-demo/main.v` (new example)

The actual vframes-touching code — and the "best practice" reference users copy from.
Self-contained (no external data file):

1. `mut ctx := vframes.init()`, `df := ctx.read_records(records)!` (or `read_csv`) over
   a small inline synthetic dataset (e.g. 20 rows, 2 features + 1 continuous target) —
   avoids depending on an external data file.
2. Conversion glue (the few lines being documented as the pattern): call
   `data := df.values()!`, match the `[]map[string]json2.Any` arm, and build
   `rows []map[string]f64` by calling `.f64()` on each value. This snippet stays
   inline in the example rather than becoming a permanent vstats function — it's the
   one place that legitimately needs to know about vframes' `Data`/`json2.Any` types,
   and it's short enough (~10 lines) not to warrant a wrapper.
3. `reg_ds := utils.rows_to_regression_dataset(rows, ['x1', 'x2'], 'y', 'synthetic')!`
4. Run a real fit: `model := ml.linear_regression(reg_ds.features, reg_ds.target)` and
   print the coefficients — proves the round trip actually works end-to-end, not just
   that it compiles.

**Modules used:** `vframes` (external, this example only), `vstats.utils`,
`vstats.ml`. README documents that this example requires vframes installed
(`v install https://github.com/rodabt/vframes`) — every other example remains
runnable with zero extra installs, and `make test` never touches this directory since
it contains no `_test.v` files.

## Non-goals

- No wrapper functions for vframes' own API surface (`filter`, `group_by`, `join`,
  etc.) — those are vframes' job; vstats only needs the landing point.
- No support for `[]map[string]string` (vframes' `as_string` output mode) — numeric
  analysis needs numeric input; string-typed columns are out of scope for this bridge.
- No change to `v.mod` — `dependencies: []` stays exactly as it is.

## Testing Strategy

- `tests/tabular_test.v` covers all 4 new functions per Part 2 above — runs under the
  existing `make test` with no new setup.
- `examples/vframes-integration-demo/main.v` is verified by running it
  (`v run examples/vframes-integration-demo/main.v`) and confirming it produces a
  sensible fitted model — this is the only place vframes is exercised, and it's
  outside the automated test suite by convention (examples are `v run`, not
  `v test`).
