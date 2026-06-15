import timeseries
import math

fn test__diff_order1() {
	x := [1.0, 3.0, 6.0, 10.0]
	got := timeseries.diff(x, 1)
	assert got == [2.0, 3.0, 4.0]
}

fn test__diff_order2() {
	x := [1.0, 3.0, 6.0, 10.0, 15.0]
	got := timeseries.diff(x, 2)
	assert got == [1.0, 1.0, 1.0]
}

fn test__diff_order0() {
	x := [1.0, 2.0, 3.0]
	assert timeseries.diff(x, 0) == [1.0, 2.0, 3.0]
}

fn test__seasonal_diff() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	got := timeseries.seasonal_diff(x, 2)
	assert got == [2.0, 2.0, 2.0, 2.0]
}

fn test__undiff_roundtrip() {
	x := [1.0, 3.0, 6.0, 10.0, 15.0]
	d := 1
	diffed := timeseries.diff(x, d)
	// forecast 2 steps: pretend diffed slice is forecast
	forecast_diff := [5.0, 6.0]
	recovered := timeseries.undiff(forecast_diff, x, d)
	assert math.abs(recovered[0] - 20.0) < 0.001  // 15 + 5
	assert math.abs(recovered[1] - 26.0) < 0.001  // 20 + 6
}
