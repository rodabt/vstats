module timeseries

import math
import stats
import linalg

// diff applies order-d differencing to series x.
// Returns a slice of length x.len - d.
pub fn diff(x []f64, d int) []f64 {
	if d == 0 {
		return x.clone()
	}
	mut result := []f64{len: x.len - 1}
	for i in 1 .. x.len {
		result[i - 1] = x[i] - x[i - 1]
	}
	return diff(result, d - 1)
}

// seasonal_diff applies seasonal differencing with the given period.
// Returns a slice of length x.len - period.
pub fn seasonal_diff(x []f64, period int) []f64 {
	mut result := []f64{len: x.len - period}
	for i in period .. x.len {
		result[i - period] = x[i] - x[i - period]
	}
	return result
}

// undiff inverts d-order differencing on a forecast array.
// original is the full pre-forecast series (needed for seed values).
// Returns forecast values in the original (undifferenced) scale.
pub fn undiff(forecast_diff []f64, original []f64, d int) []f64 {
	if d == 0 {
		return forecast_diff.clone()
	}
	// Seed is last value of the (d-1)-times differenced original
	orig_d1 := diff(original, d - 1)
	seed := orig_d1[orig_d1.len - 1]
	mut integrated := []f64{len: forecast_diff.len}
	integrated[0] = seed + forecast_diff[0]
	for i in 1 .. forecast_diff.len {
		integrated[i] = integrated[i - 1] + forecast_diff[i]
	}
	return undiff(integrated, original, d - 1)
}
