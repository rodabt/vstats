# vframes Tabular Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give vstats a pure-V way to consume row-oriented tabular data (the shape any
DataFrame library, including vframes, naturally produces) without vstats taking on any
new dependency, plus a working example that actually wires vframes into vstats
end-to-end as the documented best-practice pattern.

**Architecture:** Four new functions in `utils/tabular.v` convert `[]map[string]f64`
into the `[]f64` / `[][]f64` / `Dataset` / `RegressionDataset` shapes vstats' `ml`
functions already consume — zero new imports, zero vframes/json2 awareness. A separate
new example (`examples/vframes-integration-demo/main.v`) is the only place that
actually imports vframes: it reads data via vframes, converts vframes'
`[]map[string]json2.Any` into `[]map[string]f64` inline, and hands that to the new
`utils` functions to run a real `ml.linear_regression` fit.

**Tech Stack:** V language. `utils/tabular.v` and its tests use only what `utils`
already imports (nothing new). The example additionally uses `vframes` (external,
already available via `~/.vmodules/vframes` on this machine — matches how every other
example resolves `vstats.*` via the `~/.vmodules/vstats` symlink) and V's built-in
`x.json2` (vlib, not a third-party package).

## Global Constraints

- `v.mod`'s `dependencies: []` must not change — no new entry, ever.
- `utils/tabular.v` and `tests/tabular_test.v` must not import `vframes` or `x.json2`
  — only plain V (no imports needed at all, per the function signatures below).
- Tests live in `tests/`, not alongside source files, prefixed `test_`/`fn test__...()`
  — matches every existing vstats test file.
- `make test` / `v test tests/` must keep passing with zero new setup — this is the
  proof that the dependency-free guarantee holds for the new code.
- The vframes-touching code lives only in `examples/vframes-integration-demo/main.v`,
  verified by `v run`, never by `v test` (examples contain no `_test.v` files, so
  `v test tests/` never touches this directory).
- Row-major matrix order: `rows_to_matrix(rows, columns)[i][j]` = row `i`, column
  `columns[j]` — matches how `ml.linear_regression(x [][]T, y []T)` already consumes
  feature matrices.
- Missing-column and empty-`rows` errors use V's `error(...)` / `!` return convention
  already used throughout `utils/datasets.v`.

---

## File Structure

| File | Change |
|------|--------|
| `utils/tabular.v` | New. `rows_to_vector`, `rows_to_matrix`, `rows_to_dataset`, `rows_to_regression_dataset` — all pure V, reusing the existing `Dataset`/`RegressionDataset` structs from `utils/datasets.v` (same `utils` module, no import needed). |
| `tests/tabular_test.v` | New. Covers all 4 functions: happy path, missing-column error, empty-rows error, int-truncation for `rows_to_dataset`. |
| `examples/vframes-integration-demo/main.v` | New. Builds a synthetic dataset via `vframes.read_records`, converts its `Data` to `[]map[string]f64`, calls `utils.rows_to_regression_dataset`, runs `ml.linear_regression`, prints the fitted coefficients. |
| `examples/vframes-integration-demo/README.md` | New. Documents the pattern, notes vframes must be installed to run this one example (`v install https://github.com/rodabt/vframes`), and that no other vstats functionality requires it. |

Task order: `rows_to_vector`/`rows_to_matrix` first (foundational, shared
missing-column error-handling pattern), then `rows_to_dataset`/`rows_to_regression_dataset`
(which call the first two internally), then the example (which exercises everything
together against real vframes output).

---

## Task 1: `rows_to_vector` and `rows_to_matrix`

**Files:**
- Create: `utils/tabular.v`
- Test: `tests/tabular_test.v`

**Interfaces:**
- Consumes: nothing new — pure V, no imports.
- Produces: `pub fn rows_to_vector(rows []map[string]f64, column string) ![]f64` and
  `pub fn rows_to_matrix(rows []map[string]f64, columns []string) ![][]f64`. Task 2's
  `rows_to_dataset`/`rows_to_regression_dataset` call these two directly by name.

- [ ] **Step 1: Write the failing tests**

Create `tests/tabular_test.v`:

```v
import vstats.utils

fn test__rows_to_vector_extracts_column() {
	rows := [
		{'x': 1.0, 'y': 10.0},
		{'x': 2.0, 'y': 20.0},
		{'x': 3.0, 'y': 30.0},
	]
	result := utils.rows_to_vector(rows, 'y') or { panic(err.msg()) }
	assert result == [10.0, 20.0, 30.0]
}

fn test__rows_to_vector_errors_on_empty_rows() {
	rows := []map[string]f64{}
	utils.rows_to_vector(rows, 'y') or {
		assert err.msg().contains('empty')
		return
	}
	assert false, 'expected an error for empty rows'
}

fn test__rows_to_vector_errors_on_missing_column() {
	rows := [
		{'x': 1.0, 'y': 10.0},
		{'x': 2.0},
	]
	utils.rows_to_vector(rows, 'y') or {
		assert err.msg().contains('y')
		return
	}
	assert false, 'expected an error for a missing column'
}

fn test__rows_to_matrix_builds_row_major() {
	rows := [
		{'x1': 1.0, 'x2': 2.0},
		{'x1': 3.0, 'x2': 4.0},
	]
	result := utils.rows_to_matrix(rows, ['x1', 'x2']) or { panic(err.msg()) }
	assert result == [[1.0, 2.0], [3.0, 4.0]]
}

fn test__rows_to_matrix_errors_on_empty_rows() {
	rows := []map[string]f64{}
	utils.rows_to_matrix(rows, ['x1']) or {
		assert err.msg().contains('empty')
		return
	}
	assert false, 'expected an error for empty rows'
}

fn test__rows_to_matrix_errors_on_missing_column() {
	rows := [
		{'x1': 1.0, 'x2': 2.0},
		{'x1': 3.0},
	]
	utils.rows_to_matrix(rows, ['x1', 'x2']) or {
		assert err.msg().contains('x2')
		return
	}
	assert false, 'expected an error for a missing column'
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `v test tests/tabular_test.v`
Expected: FAIL — compilation error, `rows_to_vector`/`rows_to_matrix` not defined in
module `utils`.

- [ ] **Step 3: Write the implementation**

Create `utils/tabular.v`:

```v
module utils

// rows_to_vector extracts a single column from row-oriented data into a flat []f64,
// preserving row order. Any DataFrame library's row-map output (e.g. vframes'
// DataFrame.values()!, converted to []map[string]f64) can feed this directly.
pub fn rows_to_vector(rows []map[string]f64, column string) ![]f64 {
	if rows.len == 0 {
		return error('rows_to_vector: rows must not be empty')
	}
	mut result := []f64{cap: rows.len}
	for i, row in rows {
		if column !in row {
			return error('rows_to_vector: column "${column}" missing in row ${i}')
		}
		result << row[column]
	}
	return result
}

// rows_to_matrix extracts multiple columns from row-oriented data into a row-major
// [][]f64 -- result[i][j] is row i, column columns[j] -- matching the shape ml
// functions like linear_regression(x [][]T, y []T) already consume.
pub fn rows_to_matrix(rows []map[string]f64, columns []string) ![][]f64 {
	if rows.len == 0 {
		return error('rows_to_matrix: rows must not be empty')
	}
	if columns.len == 0 {
		return error('rows_to_matrix: columns must not be empty')
	}
	mut result := [][]f64{cap: rows.len}
	for i, row in rows {
		mut r := []f64{cap: columns.len}
		for col in columns {
			if col !in row {
				return error('rows_to_matrix: column "${col}" missing in row ${i}')
			}
			r << row[col]
		}
		result << r
	}
	return result
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `v test tests/tabular_test.v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `v test tests/`
Expected: PASS — every pre-existing test still passes; no new setup required.

- [ ] **Step 6: Commit**

```bash
git add utils/tabular.v tests/tabular_test.v
git commit -m "feat(utils): add rows_to_vector/rows_to_matrix tabular bridge functions"
```

---

## Task 2: `rows_to_dataset` and `rows_to_regression_dataset`

**Files:**
- Modify: `utils/tabular.v`
- Modify: `tests/tabular_test.v`

**Interfaces:**
- Consumes: `rows_to_vector`, `rows_to_matrix` (Task 1, same file/module, called
  directly by name). `Dataset` and `RegressionDataset` structs from
  `utils/datasets.v:6-25` (same `utils` module, no import needed).
- Produces: `pub fn rows_to_dataset(rows []map[string]f64, feature_cols []string, target_col string, name string) !Dataset`
  and `pub fn rows_to_regression_dataset(rows []map[string]f64, feature_cols []string, target_col string, name string) !RegressionDataset`.
  Task 3's example calls `rows_to_regression_dataset` directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/tabular_test.v`:

```v
fn test__rows_to_dataset_truncates_target_to_int() {
	rows := [
		{'x': 1.0, 'label': 2.9},
		{'x': 2.0, 'label': 0.4},
	]
	ds := utils.rows_to_dataset(rows, ['x'], 'label', 'synthetic') or { panic(err.msg()) }
	assert ds.target == [2, 0]
	assert ds.features == [[1.0], [2.0]]
	assert ds.feature_names == ['x']
	assert ds.target_name == 'label'
	assert ds.name == 'synthetic'
}

fn test__rows_to_dataset_propagates_missing_column_error() {
	rows := [
		{'x': 1.0},
	]
	utils.rows_to_dataset(rows, ['x'], 'label', 'synthetic') or {
		assert err.msg().contains('label')
		return
	}
	assert false, 'expected an error for a missing target column'
}

fn test__rows_to_regression_dataset_keeps_target_as_f64() {
	rows := [
		{'x': 1.0, 'y': 2.9},
		{'x': 2.0, 'y': 0.4},
	]
	ds := utils.rows_to_regression_dataset(rows, ['x'], 'y', 'synthetic') or { panic(err.msg()) }
	assert ds.target == [2.9, 0.4]
	assert ds.features == [[1.0], [2.0]]
	assert ds.feature_names == ['x']
	assert ds.target_name == 'y'
	assert ds.name == 'synthetic'
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `v test tests/tabular_test.v`
Expected: FAIL — compilation error, `rows_to_dataset`/`rows_to_regression_dataset` not
defined in module `utils`.

- [ ] **Step 3: Write the implementation**

Append to `utils/tabular.v`:

```v
// rows_to_dataset builds a classification Dataset from row-oriented data. The target
// column's values are truncated to int (V's int(v) cast) since Dataset.target is
// []int -- pass already-integer-valued columns (e.g. encoded class labels).
pub fn rows_to_dataset(rows []map[string]f64, feature_cols []string, target_col string, name string) !Dataset {
	features := rows_to_matrix(rows, feature_cols)!
	target_f64 := rows_to_vector(rows, target_col)!
	mut target := []int{cap: target_f64.len}
	for v in target_f64 {
		target << int(v)
	}
	return Dataset{
		name:          name
		features:      features
		target:        target
		feature_names: feature_cols
		target_name:   target_col
		description:   ''
	}
}

// rows_to_regression_dataset builds a RegressionDataset from row-oriented data. The
// target column stays f64 -- no cast.
pub fn rows_to_regression_dataset(rows []map[string]f64, feature_cols []string, target_col string, name string) !RegressionDataset {
	features := rows_to_matrix(rows, feature_cols)!
	target := rows_to_vector(rows, target_col)!
	return RegressionDataset{
		name:          name
		features:      features
		target:        target
		feature_names: feature_cols
		target_name:   target_col
		description:   ''
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `v test tests/tabular_test.v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `v test tests/`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add utils/tabular.v tests/tabular_test.v
git commit -m "feat(utils): add rows_to_dataset/rows_to_regression_dataset tabular bridge functions"
```

---

## Task 3: vframes integration example

**Files:**
- Create: `examples/vframes-integration-demo/main.v`
- Create: `examples/vframes-integration-demo/README.md`

**Interfaces:**
- Consumes: `utils.rows_to_regression_dataset` (Task 2), `ml.linear_regression[T](x [][]T, y []T) LinearModel[T]`
  (existing, `ml/regression.v:39`), `vframes.init()`, `ctx.read_records(dict []map[string]json2.Any) !DataFrame`
  (existing, `vframes/src/io.v:32`), `df.values(vp ValuesParams) !Data` where
  `Data = []map[string]json2.Any | []map[string]string` (existing, `vframes/src/explore.v:84`),
  and `json2.Any.f64() f64` (vlib, `x/json2/json2.v:189`).
- Produces: nothing consumed by later tasks — this is the final task.

- [ ] **Step 1: Confirm vframes is resolvable**

Run: `ls -la ~/.vmodules/vframes`
Expected: a symlink to `/home/rabt/devel/vframes` (already present on this machine —
matches how `~/.vmodules/vstats` already resolves every other example's
`import vstats.*`). If it is missing, run `v install https://github.com/rodabt/vframes`
before continuing.

- [ ] **Step 2: Write the example**

Create `examples/vframes-integration-demo/main.v`:

```v
module main

import x.json2
import vframes
import vstats.utils
import vstats.ml

fn main() {
	println('=== vframes Integration Demo ===\n')

	// Self-contained synthetic dataset: y = 3*x1 + 2*x2 + noise-free, so the fit is
	// easy to sanity-check by eye.
	mut records := []map[string]json2.Any{}
	for i in 0 .. 20 {
		x1 := f64(i)
		x2 := f64(i) * 0.5
		y := 3.0 * x1 + 2.0 * x2
		records << {
			'x1': json2.Any(x1)
			'x2': json2.Any(x2)
			'y':  json2.Any(y)
		}
	}

	mut ctx := vframes.init()
	df := ctx.read_records(records) or { panic(err) }

	// The conversion glue: vframes' Data is []map[string]json2.Any | []map[string]string.
	// Match the json2.Any arm and build the plain []map[string]f64 that vstats.utils
	// expects -- this is the entire "bridge" a caller needs to write by hand.
	data := df.values(to_stdout: false) or { panic(err) }
	mut rows := []map[string]f64{}
	match data {
		[]map[string]json2.Any {
			for record in data {
				mut row := map[string]f64{}
				for k, v in record {
					row[k] = v.f64()
				}
				rows << row
			}
		}
		[]map[string]string {
			panic('expected numeric records, got string-typed DataFrame output')
		}
	}

	reg_ds := utils.rows_to_regression_dataset(rows, ['x1', 'x2'], 'y', 'synthetic') or {
		panic(err)
	}
	model := ml.linear_regression(reg_ds.features, reg_ds.target)

	println('Fitted model: y = ${model.intercept:.4f} + ${model.coefficients[0]:.4f}*x1 + ${model.coefficients[1]:.4f}*x2')
	println('Expected:     y = 0.0000 + 3.0000*x1 + 2.0000*x2')
}
```

- [ ] **Step 3: Run it**

Run: `v run examples/vframes-integration-demo/main.v`
Expected: prints the two lines above, with fitted coefficients numerically close to
`3.0000` and `2.0000` (this is a noise-free linear relationship, so OLS should recover
the true coefficients almost exactly, e.g. `2.9999`/`2.0001` range from floating-point
rounding — not exact `3.0000` due to floating-point arithmetic, but visibly close).

- [ ] **Step 4: Write the README**

Create `examples/vframes-integration-demo/README.md`:

```markdown
# vframes Integration Demo

Demonstrates the best-practice pattern for using [vframes](https://github.com/rodabt/vframes)
(a DataFrame library for V, backed by DuckDB) as vstats' data-loading layer, **without**
vstats itself depending on vframes. vstats' `v.mod` stays `dependencies: []`; the
`utils.rows_to_vector` / `rows_to_matrix` / `rows_to_dataset` / `rows_to_regression_dataset`
functions are pure V and know nothing about vframes.

This example is the one place vframes is actually imported. It builds a small synthetic
dataset, loads it into a vframes `DataFrame`, converts vframes' row-oriented
`[]map[string]json2.Any` output into the plain `[]map[string]f64` shape
`utils.rows_to_regression_dataset` expects (about 10 lines of glue -- copy it into your
own code), and runs a real `vstats.ml.linear_regression` fit on the result.

**Run:** `v run examples/vframes-integration-demo/main.v`

**Requires:** `v install https://github.com/rodabt/vframes` (this is the *only* vstats
example that needs anything beyond vstats itself -- every other example, and
`make test`, runs with zero extra installs).

**Modules used:** `vframes`, `vstats.utils`, `vstats.ml`
```

- [ ] **Step 5: Run the full test suite one more time**

Run: `v test tests/`
Expected: PASS — confirms the example (which is never compiled by `v test`, since it
has no `_test.v` file) didn't change anything about the core dependency-free test run.

- [ ] **Step 6: Commit**

```bash
git add examples/vframes-integration-demo/
git commit -m "docs(examples): add vframes integration demo"
```

---

## Self-Review Notes

- **Spec coverage:** Part 1 (`utils/tabular.v`, all 4 functions) → Tasks 1-2. Part 2
  (`tests/tabular_test.v`) → Tasks 1-2 (same file, built incrementally). Part 3
  (`examples/vframes-integration-demo`) → Task 3. Non-goals (no vframes API wrappers,
  no `[]map[string]string` support, no `v.mod` change) are respected — no task
  introduces any of them.
- **Type consistency:** `rows_to_dataset`/`rows_to_regression_dataset` (Task 2) call
  `rows_to_matrix`/`rows_to_vector` (Task 1) with the exact signatures Task 1 defines
  (`rows []map[string]f64, columns []string` / `rows []map[string]f64, column string`).
  The example (Task 3) calls `utils.rows_to_regression_dataset` with the exact 4-arg
  signature Task 2 defines (`rows, feature_cols, target_col, name`).
- **Dependency-free guarantee:** verified structurally — `utils/tabular.v` has no
  `import` line at all (Task 1/2 code blocks confirm this), so `utils` package
  compilation is unaffected for every existing caller. Only Task 3's file imports
  `vframes`/`x.json2`, and that file is never part of `v test tests/`.
