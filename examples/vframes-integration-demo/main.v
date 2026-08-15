module main

import x.json2
import vframes
import vstats.utils
import vstats.ml

fn main() {
	println('=== vframes Integration Demo ===\n')

	// Self-contained synthetic dataset: y = 3*x1 + 2*x2 + noise-free, so the fit is
	// easy to sanity-check by eye. x2 is deliberately NOT a scalar multiple of x1
	// (it wraps via modulo) so the two features aren't collinear -- OLS needs
	// independent variation in each feature to recover both coefficients uniquely.
	mut records := []map[string]json2.Any{}
	for i in 0 .. 20 {
		x1 := f64(i)
		x2 := f64((i * 7) % 13)
		y := 3.0 * x1 + 2.0 * x2
		records << {
			'x1': json2.Any(x1)
			'x2': json2.Any(x2)
			'y':  json2.Any(y)
		}
	}

	mut ctx := vframes.init() or { panic(err) }
	df := ctx.read_records(records) or { panic(err) }

	// The conversion glue: vframes' to_dict() returns []map[string]json2.Any. Build the
	// plain []map[string]f64 that vstats.utils expects -- this is the entire "bridge"
	// a caller needs to write by hand.
	data := df.to_dict() or { panic(err) }
	mut rows := []map[string]f64{}
	for record in data {
		mut row := map[string]f64{}
		for k, v in record {
			row[k] = v.f64()
		}
		rows << row
	}

	reg_ds := utils.rows_to_regression_dataset(rows, ['x1', 'x2'], 'y', 'synthetic') or {
		panic(err)
	}
	model := ml.linear_regression(reg_ds.features, reg_ds.target)

	println('Fitted model: y = ${model.intercept:.4f} + ${model.coefficients[0]:.4f}*x1 + ${model.coefficients[1]:.4f}*x2')
	println('Expected:     y = 0.0000 + 3.0000*x1 + 2.0000*x2')
}
