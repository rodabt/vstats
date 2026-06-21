module main

import experiment
import rand

fn test__mann_whitney_clear_separation() {
	rand.seed([u32(42), u32(0)])
	ctrl := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt  := [6.0, 7.0, 8.0, 9.0, 10.0]
	r := experiment.mann_whitney_ab_test(ctrl, trt)
	assert r.significant == true
	assert r.rank_biserial > 0.9
	assert r.ci_lower > 0.0
	assert r.p_value < 0.05
	assert r.n_control == 5
	assert r.n_treatment == 5
}

fn test__mann_whitney_no_effect() {
	rand.seed([u32(42), u32(0)])
	ctrl := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt  := [1.0, 2.0, 3.0, 4.0, 5.0]
	r := experiment.mann_whitney_ab_test(ctrl, trt)
	assert r.significant == false
}

fn test__mann_whitney_one_sided_greater() {
	rand.seed([u32(42), u32(0)])
	ctrl := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt  := [6.0, 7.0, 8.0, 9.0, 10.0]
	r_two := experiment.mann_whitney_ab_test(ctrl, trt)
	rand.seed([u32(42), u32(0)])
	r_one := experiment.mann_whitney_ab_test(ctrl, trt, alternative: .greater)
	assert r_one.p_value < r_two.p_value
	assert r_one.rank_biserial == r_two.rank_biserial
}

fn test__mann_whitney_ci_contains_r() {
	rand.seed([u32(42), u32(0)])
	ctrl := [1.0, 2.0, 3.0, 4.0, 5.0]
	trt  := [6.0, 7.0, 8.0, 9.0, 10.0]
	r := experiment.mann_whitney_ab_test(ctrl, trt)
	assert r.ci_lower <= r.rank_biserial
	assert r.rank_biserial <= r.ci_upper
}
