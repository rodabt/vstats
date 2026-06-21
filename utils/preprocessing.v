module utils

pub fn winsorize(x []f64, lower f64, upper f64) []f64 {
	assert x.len >= 2, 'winsorize requires at least 2 elements'
	assert lower >= 0.0 && upper <= 1.0 && lower < upper, 'lower must be in [0,1), upper in (0,1], lower < upper'
	mut sorted := x.clone()
	sorted.sort()
	lo_val := sorted[int(lower * f64(x.len - 1))]
	hi_val := sorted[int(upper * f64(x.len - 1))]
	return x.map(if it < lo_val { lo_val } else if it > hi_val { hi_val } else { it })
}

pub fn trim(x []f64, lower f64, upper f64) []f64 {
	assert x.len >= 2, 'trim requires at least 2 elements'
	assert lower >= 0.0 && upper <= 1.0 && lower < upper, 'lower must be in [0,1), upper in (0,1], lower < upper'
	mut sorted := x.clone()
	sorted.sort()
	lo_val := sorted[int(lower * f64(x.len - 1))]
	hi_val := sorted[int(upper * f64(x.len - 1))]
	return x.filter(it >= lo_val && it <= hi_val)
}
