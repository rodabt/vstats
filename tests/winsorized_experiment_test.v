module main

import experiment

fn test__winsorized_abtest_reduces_outlier_effect() {
	ctrl := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt  := [1.0, 2.0, 3.0, 4.0, 1000.0]
	raw  := experiment.abtest(ctrl, trt)
	wins := experiment.winsorized_abtest(ctrl, trt, 0.05, 0.95)
	assert wins.treatment_mean < raw.treatment_mean
}

fn test__trimmed_abtest_reduces_outlier_effect() {
	ctrl    := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt     := [1.0, 2.0, 3.0, 4.0, 1000.0]
	raw     := experiment.abtest(ctrl, trt)
	trimmed := experiment.trimmed_abtest(ctrl, trt, 0.05, 0.95)
	assert trimmed.treatment_mean < raw.treatment_mean
}

fn test__winsorized_abtest_returns_abtestresult() {
	ctrl := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt  := [1.5, 2.5, 3.5, 4.5, 5.5]
	r    := experiment.winsorized_abtest(ctrl, trt, 0.05, 0.95)
	assert r.p_value > 0.0
	assert r.treatment_mean > 0.0
	assert r.control_mean > 0.0
}

fn test__trimmed_abtest_returns_abtestresult() {
	ctrl := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt  := [1.5, 2.5, 3.5, 4.5, 5.5]
	r    := experiment.trimmed_abtest(ctrl, trt, 0.05, 0.95)
	assert r.p_value > 0.0
	assert r.treatment_mean > 0.0
	assert r.control_mean > 0.0
}
