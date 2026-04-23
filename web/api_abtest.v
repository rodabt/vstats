module main

import veb
import json
import math
import vstats.experiment

struct ABTestRequest {
pub mut:
	metric       string
	// proportion mode
	successes_a  int
	n_a          int
	successes_b  int
	n_b          int
	// continuous raw mode
	control_data   []f64
	treatment_data []f64
	// continuous summary mode
	control_mean   f64
	control_std    f64
	control_n      int
	treatment_mean f64
	treatment_std  f64
	treatment_n    int
	alpha          f64 = 0.05
}

// Inverse normal CDF approximation (Abramowitz & Stegun 26.2.17), accurate to ±0.00045
fn z_critical(alpha f64) f64 {
	p := 1.0 - alpha / 2.0
	t := math.sqrt(-2.0 * math.log(1.0 - p))
	num := 2.515517 + 0.802853 * t + 0.010328 * t * t
	den := 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
	return t - num / den
}

struct ABTestResponse {
pub:
	metric        string
	p_value       f64
	significant   bool
	relative_lift f64
	ci_lower      f64
	ci_upper      f64
	// continuous
	t_statistic   f64
	effect_size   f64
	// proportion
	z_statistic   f64
	rate_a        f64
	rate_b        f64
}

@['/api/ab-test'; post]
pub fn (app &App) api_ab_test(mut ctx Context) veb.Result {
	req := json.decode(ABTestRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.alpha <= 0 || req.alpha >= 1 {
		return api_error(mut ctx, 'alpha must be between 0 and 1')
	}

	if req.metric == 'proportion' {
		if req.n_a < 2 || req.n_b < 2 {
			return api_error(mut ctx, 'group sizes must be at least 2')
		}
		if req.successes_a < 0 || req.successes_a > req.n_a {
			return api_error(mut ctx, 'successes_a out of range')
		}
		if req.successes_b < 0 || req.successes_b > req.n_b {
			return api_error(mut ctx, 'successes_b out of range')
		}
		r := experiment.proportion_test(req.successes_a, req.n_a, req.successes_b, req.n_b,
			experiment.ProportionTestConfig{ alpha: req.alpha })
		return ctx.json(ABTestResponse{
			metric:        'proportion'
			p_value:       r.p_value
			significant:   r.significant
			relative_lift: r.relative_lift
			ci_lower:      r.ci_lower
			ci_upper:      r.ci_upper
			z_statistic:   r.z_statistic
			rate_a:        r.rate_a
			rate_b:        r.rate_b
		})
	}

	if req.metric == 'continuous' {
		if req.control_data.len < 2 || req.treatment_data.len < 2 {
			return api_error(mut ctx, 'each group needs at least 2 observations')
		}
		r := experiment.abtest(req.control_data, req.treatment_data,
			experiment.ABTestConfig{ alpha: req.alpha })
		return ctx.json(ABTestResponse{
			metric:        'continuous'
			p_value:       r.p_value
			significant:   r.significant
			relative_lift: r.relative_lift
			ci_lower:      r.ci_lower
			ci_upper:      r.ci_upper
			t_statistic:   r.t_statistic
			effect_size:   r.effect_size
		})
	}

	if req.metric == 'summary' {
		if req.control_n < 2 || req.treatment_n < 2 {
			return api_error(mut ctx, 'each group needs at least 2 observations')
		}
		if req.control_std < 0 || req.treatment_std < 0 {
			return api_error(mut ctx, 'standard deviations must be non-negative')
		}
		na := f64(req.control_n)
		nb := f64(req.treatment_n)
		ma := req.control_mean
		mb := req.treatment_mean
		sa := req.control_std
		sb := req.treatment_std

		var_a := sa * sa / na
		var_b := sb * sb / nb
		se := math.sqrt(var_a + var_b)
		if se == 0 {
			return api_error(mut ctx, 'standard error is zero: both groups have identical variance and means cannot differ')
		}

		diff := mb - ma
		t_stat := diff / se
		p_val := 1.0 - math.erf(math.abs(t_stat) / math.sqrt(2.0))

		z_crit := z_critical(req.alpha)
		ci_lower := diff - z_crit * se
		ci_upper := diff + z_crit * se

		rel_lift := if ma != 0 { diff / math.abs(ma) } else { 0.0 }
		pooled_std := math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
		effect_size := if pooled_std > 0 { diff / pooled_std } else { 0.0 }

		return ctx.json(ABTestResponse{
			metric:        'summary'
			p_value:       p_val
			significant:   p_val < req.alpha
			relative_lift: rel_lift
			ci_lower:      ci_lower
			ci_upper:      ci_upper
			t_statistic:   t_stat
			effect_size:   effect_size
		})
	}

	return api_error(mut ctx, 'metric must be "continuous", "summary", or "proportion"')
}
