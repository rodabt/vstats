module main

import stats
import math
import rand

fn test__bootstrap_ci_estimate_matches_stat() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	rand.seed([u32(42), u32(0)])
	r := stats.bootstrap_ci(x, fn (s []f64) f64 { return stats.mean(s) }, 1000, 0.05)
	assert math.abs(r.estimate - stats.mean(x)) < 1e-10
}

fn test__bootstrap_ci_has_positive_width() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	rand.seed([u32(42), u32(0)])
	r := stats.bootstrap_ci(x, fn (s []f64) f64 { return stats.mean(s) }, 1000, 0.05)
	assert r.ci_lower < r.ci_upper
}

fn test__bootstrap_ci_median() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	rand.seed([u32(42), u32(0)])
	r := stats.bootstrap_ci(x, fn (s []f64) f64 { return stats.median(s) }, 1000, 0.05)
	assert math.abs(r.estimate - 5.5) < 1e-10
	assert r.ci_lower > 0.0
	assert r.ci_upper < 11.0
}

fn test__bootstrap_ci_clearly_positive() {
	x := [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
	rand.seed([u32(42), u32(0)])
	r := stats.bootstrap_ci(x, fn (s []f64) f64 { return stats.mean(s) }, 1000, 0.05)
	assert r.ci_lower > 0.0
}

fn test__bootstrap_ci_custom_statistic() {
	x := [1.0, 2.0, 3.0, 4.0, 5.0]
	rand.seed([u32(42), u32(0)])
	r := stats.bootstrap_ci(x, fn (s []f64) f64 {
		mut m := s[0]
		for v in s { if v > m { m = v } }
		return m
	}, 200, 0.05)
	assert r.estimate == 5.0
	assert r.ci_lower <= r.estimate
}
