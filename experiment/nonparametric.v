module experiment

import rand
import hypothesis

@[params]
pub struct NonParametricConfig {
pub:
	alpha       f64 = 0.05
	alternative TestAlternative = .two_sided
	n_boot      int = 1000
}

pub struct MannWhitneyResult {
pub:
	u_statistic   f64
	p_value       f64
	rank_biserial f64
	ci_lower      f64
	ci_upper      f64
	significant   bool
	n_control     int
	n_treatment   int
}

pub struct WilcoxonResult {
pub:
	w_statistic   f64
	p_value       f64
	rank_biserial f64
	ci_lower      f64
	ci_upper      f64
	significant   bool
	n_pairs       int
}

pub fn mann_whitney_ab_test(ctrl []f64, trt []f64, cfg NonParametricConfig) MannWhitneyResult {
	assert ctrl.len >= 2 && trt.len >= 2, 'each group needs at least 2 observations'
	n1 := ctrl.len
	n2 := trt.len

	u, p_two := hypothesis.mann_whitney_u_test(ctrl, trt)

	expected_u := f64(n1 * n2) / 2.0
	p_val := match cfg.alternative {
		.two_sided { p_two }
		.greater   { if u < expected_u { p_two / 2.0 } else { 1.0 - p_two / 2.0 } }
		.less      { if u > expected_u { p_two / 2.0 } else { 1.0 - p_two / 2.0 } }
	}

	rank_biserial := 1.0 - 2.0 * u / f64(n1 * n2)

	mut r_boot := []f64{len: cfg.n_boot}
	for i in 0 .. cfg.n_boot {
		mut boot_ctrl := []f64{len: n1}
		mut boot_trt  := []f64{len: n2}
		for j in 0 .. n1 { boot_ctrl[j] = ctrl[rand.intn(n1) or { 0 }] }
		for j in 0 .. n2 { boot_trt[j]  = trt[rand.intn(n2) or { 0 }] }
		bu, _ := hypothesis.mann_whitney_u_test(boot_ctrl, boot_trt)
		r_boot[i] = 1.0 - 2.0 * bu / f64(n1 * n2)
	}
	r_boot.sort()
	lo := int(cfg.alpha / 2.0 * f64(cfg.n_boot))
	hi := int((1.0 - cfg.alpha / 2.0) * f64(cfg.n_boot))

	return MannWhitneyResult{
		u_statistic:   u
		p_value:       p_val
		rank_biserial: rank_biserial
		ci_lower:      r_boot[lo]
		ci_upper:      r_boot[hi]
		significant:   p_val < cfg.alpha
		n_control:     n1
		n_treatment:   n2
	}
}

pub fn wilcoxon_paired_test(ctrl []f64, trt []f64, cfg NonParametricConfig) WilcoxonResult {
	assert ctrl.len == trt.len, 'ctrl and trt must have equal length for paired test'
	assert ctrl.len >= 2, 'need at least 2 pairs'
	n := ctrl.len

	mut n_nonzero := 0
	for i in 0 .. n {
		if ctrl[i] != trt[i] { n_nonzero++ }
	}

	w_plus, p_two := hypothesis.wilcoxon_signed_rank_test(ctrl, trt)

	expected_w := f64(n_nonzero * (n_nonzero + 1)) / 4.0
	p_val := match cfg.alternative {
		.two_sided { p_two }
		.greater   { if w_plus < expected_w { p_two / 2.0 } else { 1.0 - p_two / 2.0 } }
		.less      { if w_plus > expected_w { p_two / 2.0 } else { 1.0 - p_two / 2.0 } }
	}

	w_total := f64(n_nonzero * (n_nonzero + 1)) / 2.0
	rank_biserial := if w_total > 0.0 { 1.0 - 2.0 * w_plus / w_total } else { 0.0 }

	mut r_boot := []f64{len: cfg.n_boot}
	for i in 0 .. cfg.n_boot {
		mut boot_ctrl := []f64{len: n}
		mut boot_trt  := []f64{len: n}
		for j in 0 .. n {
			idx := rand.intn(n) or { 0 }
			boot_ctrl[j] = ctrl[idx]
			boot_trt[j]  = trt[idx]
		}
		bw, _ := hypothesis.wilcoxon_signed_rank_test(boot_ctrl, boot_trt)
		mut boot_n_nonzero := 0
		for j in 0 .. n {
			if boot_ctrl[j] != boot_trt[j] { boot_n_nonzero++ }
		}
		bw_total := f64(boot_n_nonzero * (boot_n_nonzero + 1)) / 2.0
		r_boot[i] = if bw_total > 0.0 { 1.0 - 2.0 * bw / bw_total } else { 0.0 }
	}
	r_boot.sort()
	lo := int(cfg.alpha / 2.0 * f64(cfg.n_boot))
	hi := int((1.0 - cfg.alpha / 2.0) * f64(cfg.n_boot))

	return WilcoxonResult{
		w_statistic:   w_plus
		p_value:       p_val
		rank_biserial: rank_biserial
		ci_lower:      r_boot[lo]
		ci_upper:      r_boot[hi]
		significant:   p_val < cfg.alpha
		n_pairs:       n
	}
}
