module main

import veb
import json
import vstats.experiment

struct DiDRequest {
pub mut:
	method string
	alpha  f64 = 0.05
	// simple 2x2
	y_treat_pre  []f64
	y_treat_post []f64
	y_ctrl_pre   []f64
	y_ctrl_post  []f64
	// regression
	y     []f64
	x     [][]f64
	group []int
	time  []int
	// parallel trends
	y_treated_pre []f64
	y_control_pre []f64
	time_pre      []int
	// event study
	relative_time []int
}

struct DiDSimpleResponse {
pub:
	did_effect    f64
	se            f64
	t_statistic   f64
	p_value       f64
	ci_lower      f64
	ci_upper      f64
	treated_change f64
	control_change f64
}

struct DiDRegressionResponse {
pub:
	did_coefficient f64
	did_se          f64
	did_t_stat      f64
	did_p_value     f64
	did_ci_lower    f64
	did_ci_upper    f64
	r_squared       f64
	n               int
}

struct ParallelTrendsResponse {
pub:
	slope_treated        f64
	slope_control        f64
	slope_difference     f64
	t_statistic          f64
	p_value              f64
	parallel_trends_hold bool
}

struct EventStudyResponse {
pub:
	relative_times []int
	effects        []f64
	std_errors     []f64
	t_statistics   []f64
	p_values       []f64
	ci_lowers      []f64
	ci_uppers      []f64
}

@['/api/did'; post]
pub fn (app &App) api_did(mut ctx Context) veb.Result {
	req := json.decode(DiDRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	alpha := if req.alpha > 0 && req.alpha < 1 { req.alpha } else { 0.05 }
	cfg := experiment.DiDConfig{ alpha: alpha }

	match req.method {
		'simple' {
			if req.y_treat_pre.len < 2 || req.y_treat_post.len < 2
				|| req.y_ctrl_pre.len < 2 || req.y_ctrl_post.len < 2 {
				return api_error(mut ctx, 'each group/period needs at least 2 observations')
			}
			r := experiment.did_2x2(req.y_treat_pre, req.y_treat_post, req.y_ctrl_pre, req.y_ctrl_post, cfg)
			return ctx.json(DiDSimpleResponse{
				did_effect:     r.did_effect
				se:             r.se
				t_statistic:    r.t_statistic
				p_value:        r.p_value
				ci_lower:       r.ci_lower
				ci_upper:       r.ci_upper
				treated_change: r.treated_change
				control_change: r.control_change
			})
		}
		'regression' {
			n := req.y.len
			if n < 4 {
				return api_error(mut ctx, 'need at least 4 observations')
			}
			if req.group.len != n || req.time.len != n {
				return api_error(mut ctx, 'y, group, and time must have the same length')
			}
			x := if req.x.len == n { req.x } else { req.y.map([]f64{}) }
			r := experiment.did_regression(req.y, x, req.group, req.time, cfg)
			return ctx.json(DiDRegressionResponse{
				did_coefficient: r.did_coefficient
				did_se:          r.did_se
				did_t_stat:      r.did_t_stat
				did_p_value:     r.did_p_value
				did_ci_lower:    r.did_ci_lower
				did_ci_upper:    r.did_ci_upper
				r_squared:       r.r_squared
				n:               r.n
			})
		}
		'parallel' {
			if req.y_treated_pre.len < 3 || req.y_control_pre.len < 3 {
				return api_error(mut ctx, 'need at least 3 pre-period observations per group')
			}
			if req.time_pre.len != req.y_treated_pre.len || req.time_pre.len != req.y_control_pre.len {
				return api_error(mut ctx, 'time_pre must match the length of pre-period arrays')
			}
			r := experiment.test_parallel_trends(req.y_treated_pre, req.y_control_pre, req.time_pre, cfg)
			return ctx.json(ParallelTrendsResponse{
				slope_treated:        r.slope_treated
				slope_control:        r.slope_control
				slope_difference:     r.slope_difference
				t_statistic:          r.t_statistic
				p_value:              r.p_value
				parallel_trends_hold: r.parallel_trends_hold
			})
		}
		'event' {
			n := req.y.len
			if n < 4 {
				return api_error(mut ctx, 'need at least 4 observations')
			}
			if req.group.len != n || req.relative_time.len != n {
				return api_error(mut ctx, 'y, group, and relative_time must have the same length')
			}
			r := experiment.event_study(req.y, req.group, req.relative_time, cfg)
			return ctx.json(EventStudyResponse{
				relative_times: r.relative_times
				effects:        r.effects
				std_errors:     r.std_errors
				t_statistics:   r.t_statistics
				p_values:       r.p_values
				ci_lowers:      r.ci_lowers
				ci_uppers:      r.ci_uppers
			})
		}
		else {
			return api_error(mut ctx, 'method must be simple, regression, parallel, or event')
		}
	}
}
