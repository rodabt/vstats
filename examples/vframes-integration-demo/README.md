# vframes Integration Demo

Demonstrates the best-practice pattern for using [vframes](https://github.com/rodabt/vframes)
(a DataFrame library for V, backed by DuckDB) as vstats' data-loading layer, **without**
vstats itself depending on vframes. vstats' `v.mod` stays `dependencies: []`; the
`utils.rows_to_vector` / `rows_to_matrix` / `rows_to_dataset` / `rows_to_regression_dataset`
functions are pure V and know nothing about vframes.

This example is the one place vframes is actually imported. It builds a small synthetic
dataset, loads it into a vframes `DataFrame`, converts vframes' row-oriented
`[]map[string]json2.Any` output (via `to_dict()`) into the plain `[]map[string]f64` shape
`utils.rows_to_regression_dataset` expects (about 10 lines of glue -- copy it into your
own code), and runs a real `vstats.ml.linear_regression` fit on the result.

**Run:** `v run examples/vframes-integration-demo/main.v`

**Requires:** `v install https://github.com/rodabt/vframes` (this is the *only* vstats
example that needs anything beyond vstats itself -- every other example, and
`make test`, runs with zero extra installs).

**Modules used:** `vframes`, `vstats.utils`, `vstats.ml`
