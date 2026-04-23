module main

import veb
import json
import vstats.experiment

struct PowerRequest {
pub mut:
	metric      string
	// continuous
	effect_size f64
	// proportion
	p_baseline  f64
	p_treatment f64
	// shared
	alpha f64 = 0.05
	power f64 = 0.80
}

struct PowerResponse {
pub:
	n_per_group int
	power       f64
	alpha       f64
	effect_size f64
	mde         f64
}

@['/api/power-analysis'; post]
pub fn (app &App) api_power_analysis(mut ctx Context) veb.Result {
	req := json.decode(PowerRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.alpha <= 0 || req.alpha >= 1 {
		return api_error(mut ctx, 'alpha must be between 0 and 1')
	}
	if req.power <= 0 || req.power >= 1 {
		return api_error(mut ctx, 'power must be between 0 and 1')
	}

	if req.metric == 'continuous' {
		if req.effect_size <= 0 {
			return api_error(mut ctx, 'effect_size must be positive')
		}
		r := experiment.power_analysis(req.effect_size, req.alpha, req.power)
		return ctx.json(PowerResponse{
			n_per_group: r.n_per_group
			power:       r.power
			alpha:       r.alpha
			effect_size: r.effect_size
		})
	}

	if req.metric == 'proportion' {
		if req.p_baseline <= 0 || req.p_baseline >= 1 {
			return api_error(mut ctx, 'p_baseline must be between 0 and 1')
		}
		if req.p_treatment <= 0 || req.p_treatment >= 1 {
			return api_error(mut ctx, 'p_treatment must be between 0 and 1')
		}
		if req.p_baseline == req.p_treatment {
			return api_error(mut ctx, 'p_baseline and p_treatment must differ')
		}
		r := experiment.proportion_power_analysis(req.p_baseline, req.p_treatment, req.alpha, req.power)
		return ctx.json(PowerResponse{
			n_per_group: r.n_per_group
			power:       r.power
			alpha:       r.alpha
			mde:         r.mde
		})
	}

	return api_error(mut ctx, 'metric must be "continuous" or "proportion"')
}
