module main

import veb
import json
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
	alpha          f64 = 0.05
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

	return api_error(mut ctx, 'metric must be "continuous" or "proportion"')
}
