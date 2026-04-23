module main

import veb
import json
import vstats.hypothesis

struct HypothesisRequest {
pub mut:
	test        string
	x           []f64
	y           []f64
	mu          f64
	alpha       f64 = 0.05
	contingency [][]int
}

struct HypothesisResponse {
pub:
	test      string
	statistic f64
	p_value   f64
	significant bool
}

@['/api/hypothesis'; post]
pub fn (app &App) api_hypothesis(mut ctx Context) veb.Result {
	req := json.decode(HypothesisRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	alpha := if req.alpha > 0 && req.alpha < 1 { req.alpha } else { 0.05 }
	tp := hypothesis.TestParams{}

	match req.test {
		't_test_two_sample' {
			if req.x.len < 2 || req.y.len < 2 {
				return api_error(mut ctx, 'each group needs at least 2 observations')
			}
			stat, p := hypothesis.t_test_two_sample(req.x, req.y, tp)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		't_test_one_sample' {
			if req.x.len < 2 {
				return api_error(mut ctx, 'need at least 2 observations')
			}
			stat, p := hypothesis.t_test_one_sample(req.x, req.mu, tp)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'mann_whitney' {
			if req.x.len < 2 || req.y.len < 2 {
				return api_error(mut ctx, 'each group needs at least 2 observations')
			}
			stat, p := hypothesis.mann_whitney_u_test(req.x, req.y)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'ks_test' {
			if req.x.len < 2 || req.y.len < 2 {
				return api_error(mut ctx, 'each group needs at least 2 observations')
			}
			stat, p := hypothesis.ks_test(req.x, req.y)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'chi_squared' {
			if req.contingency.len < 2 {
				return api_error(mut ctx, 'contingency table must have at least 2 rows')
			}
			stat, p := hypothesis.chi_squared_test(req.contingency)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'shapiro_wilk' {
			if req.x.len < 3 {
				return api_error(mut ctx, 'need at least 3 observations')
			}
			stat, p := hypothesis.shapiro_wilk_test(req.x)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'correlation' {
			if req.x.len < 3 || req.y.len < 3 {
				return api_error(mut ctx, 'need at least 3 observations per variable')
			}
			if req.x.len != req.y.len {
				return api_error(mut ctx, 'x and y must have the same length')
			}
			stat, p := hypothesis.correlation_test(req.x, req.y, tp)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'wilcoxon' {
			if req.x.len < 2 || req.y.len < 2 {
				return api_error(mut ctx, 'need at least 2 paired observations')
			}
			if req.x.len != req.y.len {
				return api_error(mut ctx, 'x and y must have the same length for paired test')
			}
			stat, p := hypothesis.wilcoxon_signed_rank_test(req.x, req.y)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'spearman_correlation' {
			if req.x.len < 3 || req.y.len < 3 {
				return api_error(mut ctx, 'need at least 3 observations per variable')
			}
			if req.x.len != req.y.len {
				return api_error(mut ctx, 'x and y must have the same length')
			}
			stat, p := hypothesis.spearman_correlation_test(req.x, req.y, tp)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		'runs_test' {
			if req.x.len < 10 {
				return api_error(mut ctx, 'need at least 10 observations for the runs test')
			}
			stat, p := hypothesis.runs_test(req.x)
			return ctx.json(HypothesisResponse{ test: req.test, statistic: stat, p_value: p, significant: p < alpha })
		}
		else {
			return api_error(mut ctx, 'unknown test: ${req.test}')
		}
	}
}
