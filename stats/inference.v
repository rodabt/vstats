module stats

import math
import rand

@[params]
pub struct DeltaMethodConfig {
pub:
	alpha f64 = 0.05
}

pub struct DeltaMethodResult {
pub:
	ratio_ctrl  f64
	ratio_trt   f64
	effect      f64
	se          f64
	t_statistic f64
	p_value     f64
	ci_lower    f64
	ci_upper    f64
}

pub struct BootstrapResult {
pub:
	p_value       f64
	observed_diff f64
	ci_lower      f64
	ci_upper      f64
	n_resamples   int
}

// Normal CDF via error function. Used internally since stats cannot import prob.
fn norm_cdf(x f64) f64 {
	return 0.5 * (1.0 + math.erf(x / math.sqrt2))
}

// Rational approximation for inverse normal CDF (Abramowitz & Stegun 26.2.17).
// Accuracy ~0.0005. Used internally since stats cannot import prob.
fn inv_normal_cdf(p f64) f64 {
	if p <= 0.0 { return -8.0 }
	if p >= 1.0 { return 8.0 }
	mut q := p
	mut sgn := 1.0
	if p < 0.5 {
		q = 1.0 - p
		sgn = -1.0
	}
	t := math.sqrt(-2.0 * math.log(1.0 - q))
	x := t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
		(1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
	return sgn * x
}

pub fn delta_method_ratio(a []f64, b []f64, treatment []int, cfg DeltaMethodConfig) DeltaMethodResult {
	assert a.len == b.len && a.len == treatment.len, 'a, b, treatment must have same length'
	assert a.len >= 4, 'need at least 4 observations'

	n := a.len
	mut sum_a := 0.0
	mut sum_b := 0.0
	for i in 0 .. n {
		sum_a += a[i]
		sum_b += b[i]
	}
	r := if sum_b != 0.0 { sum_a / sum_b } else { 0.0 }

	// Linearize: z_i = a_i - R * b_i
	mut z_ctrl := []f64{}
	mut z_trt  := []f64{}
	mut a_ctrl := []f64{}
	mut a_trt  := []f64{}
	mut b_ctrl := []f64{}
	mut b_trt  := []f64{}
	for i in 0 .. n {
		z := a[i] - r * b[i]
		if treatment[i] == 0 {
			z_ctrl << z
			a_ctrl << a[i]
			b_ctrl << b[i]
		} else {
			z_trt << z
			a_trt << a[i]
			b_trt << b[i]
		}
	}
	assert z_ctrl.len >= 2 && z_trt.len >= 2, 'each group needs at least 2 observations'

	mean_b := sum_b / f64(n)
	v_c := variance(z_ctrl)
	v_t := variance(z_trt)
	n_c := f64(z_ctrl.len)
	n_t := f64(z_trt.len)
	se_z := math.sqrt(v_c / n_c + v_t / n_t)
	se_ratio := if mean_b != 0.0 { se_z / math.abs(mean_b) } else { 0.0 }

	ratio_ctrl := if mean(b_ctrl) != 0.0 { mean(a_ctrl) / mean(b_ctrl) } else { 0.0 }
	ratio_trt  := if mean(b_trt)  != 0.0 { mean(a_trt)  / mean(b_trt)  } else { 0.0 }
	effect := ratio_trt - ratio_ctrl
	t_stat := if se_ratio > 0.0 { effect / se_ratio } else { 0.0 }
	p_val := 2.0 * norm_cdf(-math.abs(t_stat))
	z_crit := inv_normal_cdf(1.0 - cfg.alpha / 2.0)

	return DeltaMethodResult{
		ratio_ctrl:  ratio_ctrl
		ratio_trt:   ratio_trt
		effect:      effect
		se:          se_ratio
		t_statistic: t_stat
		p_value:     p_val
		ci_lower:    effect - z_crit * se_ratio
		ci_upper:    effect + z_crit * se_ratio
	}
}

pub fn bootstrap_test(ctrl []f64, trt []f64, n_resamples int) BootstrapResult {
	assert ctrl.len >= 2 && trt.len >= 2, 'each group needs at least 2 observations'
	assert n_resamples > 0, 'n_resamples must be positive'

	n_c := ctrl.len
	n_t := trt.len
	observed_diff := mean(trt) - mean(ctrl)

	// Permutation test: pool, shuffle, split, count extreme diffs
	mut pooled := []f64{}
	pooled << ctrl
	pooled << trt
	n_pool := pooled.len

	mut count_extreme := 0
	for _ in 0 .. n_resamples {
		mut perm := pooled.clone()
		for i := n_pool - 1; i > 0; i-- {
			j := rand.intn(i + 1) or { 0 }
			tmp := perm[i]
			perm[i] = perm[j]
			perm[j] = tmp
		}
		mut s_ctrl := 0.0
		mut s_trt  := 0.0
		for k in 0 .. n_c { s_ctrl += perm[k] }
		for k in n_c .. n_pool { s_trt += perm[k] }
		perm_diff := s_trt / f64(n_t) - s_ctrl / f64(n_c)
		if math.abs(perm_diff) >= math.abs(observed_diff) {
			count_extreme++
		}
	}
	p_val := f64(count_extreme) / f64(n_resamples)

	// Percentile bootstrap CI (resample with replacement from each group)
	mut boot_diffs := []f64{len: n_resamples}
	for i in 0 .. n_resamples {
		mut bc_sum := 0.0
		mut bt_sum := 0.0
		for _ in 0 .. n_c {
			bc_sum += ctrl[rand.intn(n_c) or { 0 }]
		}
		for _ in 0 .. n_t {
			bt_sum += trt[rand.intn(n_t) or { 0 }]
		}
		boot_diffs[i] = bt_sum / f64(n_t) - bc_sum / f64(n_c)
	}
	boot_diffs.sort()
	ci_lo := quantile(boot_diffs, 0.025)
	ci_hi := quantile(boot_diffs, 0.975)

	return BootstrapResult{
		p_value:       p_val
		observed_diff: observed_diff
		ci_lower:      ci_lo
		ci_upper:      ci_hi
		n_resamples:   n_resamples
	}
}
