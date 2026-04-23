module main

import veb
import json
import vstats.experiment

struct PSMRequest {
pub mut:
	treatment  []f64
	y          []f64
	x          [][]f64
	caliper    f64 = -1.0
	iterations int = 1000
}

struct PSMMatchingResponse {
pub:
	n_matched_treated   int
	n_unmatched_treated int
	avg_distance        f64
}

struct PSMBalanceResponse {
pub:
	mean_abs_smd_before f64
	mean_abs_smd_after  f64
	balanced            bool
}

struct PSMATEResponse {
pub:
	ate         f64
	se          f64
	ci_lower    f64
	ci_upper    f64
	t_statistic f64
	p_value     f64
	n_treated   int
	n_control   int
}

struct PSMResponse {
pub:
	matching PSMMatchingResponse
	balance  PSMBalanceResponse
	ate      PSMATEResponse
}

@['/api/psm'; post]
pub fn (app &App) api_psm(mut ctx Context) veb.Result {
	req := json.decode(PSMRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	n := req.treatment.len
	if n < 4 {
		return api_error(mut ctx, 'need at least 4 units')
	}
	if req.y.len != n {
		return api_error(mut ctx, 'y and treatment must have the same length')
	}
	if req.x.len != n {
		return api_error(mut ctx, 'x must have one row per unit')
	}
	for v in req.treatment {
		if v != 0.0 && v != 1.0 {
			return api_error(mut ctx, 'treatment must be binary (0 or 1)')
		}
	}

	model := experiment.estimate_propensity_scores(req.x, req.treatment,
		experiment.PropensityConfig{ iterations: req.iterations, learning_rate: 0.1 })

	matching := experiment.match_nearest_neighbor(model,
		experiment.MatchingConfig{ caliper: req.caliper, replacement: true })

	balance := experiment.check_balance(req.x, req.treatment, matching)
	ate := experiment.ate_matched(req.y, req.treatment, matching)

	return ctx.json(PSMResponse{
		matching: PSMMatchingResponse{
			n_matched_treated:   matching.n_matched_treated
			n_unmatched_treated: matching.n_unmatched_treated
			avg_distance:        matching.avg_distance
		}
		balance: PSMBalanceResponse{
			mean_abs_smd_before: balance.mean_abs_smd_before
			mean_abs_smd_after:  balance.mean_abs_smd_after
			balanced:            balance.balanced
		}
		ate: PSMATEResponse{
			ate:         ate.ate
			se:          ate.se
			ci_lower:    ate.ci_lower
			ci_upper:    ate.ci_upper
			t_statistic: ate.t_statistic
			p_value:     ate.p_value
			n_treated:   ate.n_treated
			n_control:   ate.n_control
		}
	})
}
