module experiment

import utils

pub fn winsorized_abtest(ctrl []f64, trt []f64, lower f64, upper f64, cfg ABTestConfig) ABTestResult {
	w_ctrl := utils.winsorize(ctrl, lower, upper)
	w_trt  := utils.winsorize(trt, lower, upper)
	return abtest(w_ctrl, w_trt, cfg)
}

pub fn trimmed_abtest(ctrl []f64, trt []f64, lower f64, upper f64, cfg ABTestConfig) ABTestResult {
	t_ctrl := utils.trim(ctrl, lower, upper)
	t_trt  := utils.trim(trt, lower, upper)
	return abtest(t_ctrl, t_trt, cfg)
}
