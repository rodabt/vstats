module main

import veb
import json
import vstats.experiment

struct CUPEDRequest {
pub mut:
	control_post   []f64
	control_pre    []f64
	treatment_post []f64
	treatment_pre  []f64
	alpha          f64 = 0.05
}

struct CUPEDAdjusted {
pub:
	p_value       f64
	significant   bool
	relative_lift f64
	t_statistic   f64
	effect_size   f64
	ci_lower      f64
	ci_upper      f64
}

struct CUPEDResponse {
pub:
	theta              f64
	variance_reduction f64
	adjusted           CUPEDAdjusted
}

@['/api/cuped'; post]
pub fn (app &App) api_cuped(mut ctx Context) veb.Result {
	req := json.decode(CUPEDRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.control_post.len < 2 || req.treatment_post.len < 2 {
		return api_error(mut ctx, 'each group needs at least 2 observations')
	}
	if req.control_post.len != req.control_pre.len {
		return api_error(mut ctx, 'control pre/post arrays must have the same length')
	}
	if req.treatment_post.len != req.treatment_pre.len {
		return api_error(mut ctx, 'treatment pre/post arrays must have the same length')
	}
	if req.alpha <= 0 || req.alpha >= 1 {
		return api_error(mut ctx, 'alpha must be between 0 and 1')
	}

	r := experiment.cuped_test(req.control_post, req.treatment_post, req.control_pre,
		req.treatment_pre, experiment.ABTestConfig{ alpha: req.alpha })

	return ctx.json(CUPEDResponse{
		theta:              r.theta
		variance_reduction: r.variance_reduction
		adjusted: CUPEDAdjusted{
			p_value:       r.adjusted_result.p_value
			significant:   r.adjusted_result.significant
			relative_lift: r.adjusted_result.relative_lift
			t_statistic:   r.adjusted_result.t_statistic
			effect_size:   r.adjusted_result.effect_size
			ci_lower:      r.adjusted_result.ci_lower
			ci_upper:      r.adjusted_result.ci_upper
		}
	})
}
