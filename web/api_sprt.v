module main

import veb
import json
import vstats.experiment

struct SPRTRequest {
pub mut:
	successes_a int
	n_a         int
	successes_b int
	n_b         int
	mde         f64
	alpha       f64 = 0.05
	beta        f64 = 0.20
}

struct SPRTResponse {
pub:
	log_likelihood_ratio f64
	decision             string
	upper_boundary       f64
	lower_boundary       f64
	rate_a               f64
	rate_b               f64
	n_a                  int
	n_b                  int
}

@['/api/sprt'; post]
pub fn (app &App) api_sprt(mut ctx Context) veb.Result {
	req := json.decode(SPRTRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.n_a < 1 || req.n_b < 1 {
		return api_error(mut ctx, 'group sizes must be positive')
	}
	if req.mde <= 0 {
		return api_error(mut ctx, 'mde must be positive')
	}
	if req.alpha <= 0 || req.alpha >= 1 {
		return api_error(mut ctx, 'alpha must be between 0 and 1')
	}
	if req.beta <= 0 || req.beta >= 1 {
		return api_error(mut ctx, 'beta must be between 0 and 1')
	}
	if req.successes_a < 0 || req.successes_a > req.n_a {
		return api_error(mut ctx, 'successes_a out of range')
	}
	if req.successes_b < 0 || req.successes_b > req.n_b {
		return api_error(mut ctx, 'successes_b out of range')
	}

	r := experiment.sprt_test(req.successes_a, req.n_a, req.successes_b, req.n_b,
		experiment.SPRTConfig{ alpha: req.alpha, beta: req.beta, mde: req.mde })

	decision := match r.decision {
		.reject_null       { 'reject_null' }
		.accept_null       { 'accept_null' }
		.continue_testing  { 'continue_testing' }
	}

	return ctx.json(SPRTResponse{
		log_likelihood_ratio: r.log_likelihood_ratio
		decision:             decision
		upper_boundary:       r.upper_boundary
		lower_boundary:       r.lower_boundary
		rate_a:               r.rate_a
		rate_b:               r.rate_b
		n_a:                  r.n_a
		n_b:                  r.n_b
	})
}
