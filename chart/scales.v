module chart

import math

pub struct LinearScale {
pub:
	domain_min f64
	domain_max f64
	range_min  f64
	range_max  f64
}

pub fn (s LinearScale) map(v f64) f64 {
	if s.domain_max == s.domain_min {
		return (s.range_min + s.range_max) / 2.0
	}
	t := (v - s.domain_min) / (s.domain_max - s.domain_min)
	return s.range_min + t * (s.range_max - s.range_min)
}

fn nice_num(x f64, round bool) f64 {
	exp := math.floor(math.log10(x))
	f := x / math.pow(10.0, exp)
	mut nf := 0.0
	if round {
		nf = if f < 1.5 {
			1.0
		} else if f < 3.0 {
			2.0
		} else if f < 7.0 {
			5.0
		} else {
			10.0
		}
	} else {
		nf = if f <= 1.0 {
			1.0
		} else if f <= 2.0 {
			2.0
		} else if f <= 5.0 {
			5.0
		} else {
			10.0
		}
	}
	return nf * math.pow(10.0, exp)
}

pub fn nice_ticks(min f64, max f64, target int) []f64 {
	if min == max {
		return [min]
	}
	span := nice_num(max - min, false)
	step := nice_num(span / f64(target - 1), true)
	graph_min := math.floor(min / step) * step
	graph_max := math.ceil(max / step) * step
	mut ticks := []f64{}
	mut v := graph_min
	for v <= graph_max + step * 0.5 {
		ticks << v
		v += step
	}
	return ticks
}

pub fn fmt_tick(v f64) string {
	if v == 0.0 {
		return '0'
	}
	if v == math.floor(v) && math.abs(v) < 1.0e15 {
		return '${i64(v)}'
	}
	return '${v:.2f}'
}
