module experiment

import math

pub struct SRMResult {
pub:
	expected_ctrl   f64
	expected_trt    f64
	observed_ctrl   int
	observed_trt    int
	chi2_statistic  f64
	p_value         f64
	srm_detected    bool
}

pub struct SimpsonsCheckResult {
pub:
	overall_effect   f64
	subgroup_effects []f64
	reversal_found   bool
}

pub struct HTEResult {
pub:
	subgroup_labels  []string
	subgroup_effects []f64
	subgroup_p_values []f64
	subgroup_ns      []int
}

// srm_test detects Sample Ratio Mismatch via a chi-squared test.
// n_ctrl, n_trt: observed counts. expected_ratio: expected trt/(ctrl+trt) split (e.g. 0.5 for 50/50).
pub fn srm_test(n_ctrl int, n_trt int, expected_ratio f64, alpha f64) SRMResult {
	assert n_ctrl > 0 && n_trt > 0, 'group counts must be positive'
	assert expected_ratio > 0.0 && expected_ratio < 1.0, 'expected_ratio must be in (0,1)'
	n_total := f64(n_ctrl + n_trt)
	exp_trt  := n_total * expected_ratio
	exp_ctrl := n_total * (1.0 - expected_ratio)
	obs_ctrl := f64(n_ctrl)
	obs_trt  := f64(n_trt)
	chi2 := (obs_ctrl - exp_ctrl) * (obs_ctrl - exp_ctrl) / exp_ctrl +
	        (obs_trt  - exp_trt)  * (obs_trt  - exp_trt)  / exp_trt
	// chi2 CDF approximation: p = 1 - chi2_sf with df=1
	// Using the regularized lower incomplete gamma: Γ(0.5, chi2/2) / Γ(0.5)
	// Equivalent to: p_val = 1 - erf(sqrt(chi2/2))
	p_val := 1.0 - math.erf(math.sqrt(chi2 / 2.0))
	return SRMResult{
		expected_ctrl:  exp_ctrl
		expected_trt:   exp_trt
		observed_ctrl:  n_ctrl
		observed_trt:   n_trt
		chi2_statistic: chi2
		p_value:        p_val
		srm_detected:   p_val < alpha
	}
}

// simpsons_check tests for Simpson's paradox: checks whether the overall effect direction
// is reversed in any subgroup.
// overall: overall treatment effect (e.g. from abtest).
// subgroup_effects: per-subgroup treatment effects (same sign convention as overall).
pub fn simpsons_check(overall_effect f64, subgroup_effects []f64) SimpsonsCheckResult {
	assert subgroup_effects.len > 0, 'subgroup_effects must not be empty'
	mut reversal := false
	for se in subgroup_effects {
		if overall_effect > 0 && se < 0 {
			reversal = true
			break
		}
		if overall_effect < 0 && se > 0 {
			reversal = true
			break
		}
	}
	return SimpsonsCheckResult{
		overall_effect:  overall_effect
		subgroup_effects: subgroup_effects
		reversal_found:  reversal
	}
}

// hte_subgroup runs Heterogeneous Treatment Effect analysis across labeled subgroups.
// y, treatment, subgroup: parallel arrays. subgroup values must be 0-based indices into labels.
// Returns per-subgroup effect, p-value, and count.
pub fn hte_subgroup(y []f64, treatment []int, subgroup []int, labels []string, cfg ABTestConfig) HTEResult {
	assert y.len == treatment.len && y.len == subgroup.len, 'y, treatment, subgroup must have same length'
	k := labels.len
	assert k > 0, 'labels must not be empty'

	mut effects   := []f64{len: k}
	mut p_values  := []f64{len: k, init: 1.0}
	mut ns        := []int{len: k}

	for sg in 0 .. k {
		mut y_ctrl := []f64{}
		mut y_trt  := []f64{}
		for i in 0 .. y.len {
			if subgroup[i] != sg { continue }
			if treatment[i] == 0 {
				y_ctrl << y[i]
			} else {
				y_trt << y[i]
			}
		}
		ns[sg] = y_ctrl.len + y_trt.len
		if y_ctrl.len >= 2 && y_trt.len >= 2 {
			res := abtest(y_ctrl, y_trt, cfg)
			effects[sg]  = res.treatment_mean - res.control_mean
			p_values[sg] = res.p_value
		}
	}

	return HTEResult{
		subgroup_labels:   labels
		subgroup_effects:  effects
		subgroup_p_values: p_values
		subgroup_ns:       ns
	}
}
