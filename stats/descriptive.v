module stats

import math
import arrays
import linalg

pub fn sum(x []f64) f64 {
	return arrays.reduce(x, fn (v1 f64, v2 f64) f64 { return v1 + v2 }) or { f64(0) }
}

pub fn mean(x []f64) f64 {
	return sum(x)/x.len
}

fn median_odd(x []f64) f64 {
	return x.sorted()[x.len / 2]
}

fn median_even(x []f64) f64 {
	x_sorted := x.sorted()
	hi_midpoint := x.len / 2 
	return (x_sorted[hi_midpoint - 1] + x_sorted[hi_midpoint]) / 2
}

pub fn median(x []f64) f64 {
	return if x.len % 2 == 0 { median_even(x) } else { median_odd(x) }
}

pub fn quantile(x []f64, p f64) f64 {
	p_index := int(p * x.len)
	return x.sorted()[p_index]
}

pub fn mode(x []f64) []f64 {
	counts := arrays.map_of_counts(x)
	max_count := arrays.max(counts.values()) or {0}
	mut res := []f64{}
	for k,v  in counts {
		if v == max_count {
			res << k
		}
	}
	return res
}


pub fn data_range(x []f64) f64 {
	return arrays.max(x) or {0} - arrays.min(x) or {0}
}

// Deviations from the mean
pub fn dev_mean(x []f64) []f64 {
	x_bar := mean(x)
	return x.map(it - x_bar)
}

// Sample variance
pub fn variance(x []f64) f64 {
	assert x.len >= 2, "variance requires at least two elements"
	n := f64(x.len)
	devs := dev_mean(x)
	return linalg.sum_of_squares(devs)/(n-1)
}

// Sample standard deviation
pub fn standard_deviation(x []f64) f64 {
	return math.sqrt(variance(x))
}

// Interquartile range
pub fn interquartile_range(x []f64) f64 {
	return quantile(x, 0.75) - quantile(x, 0.25)
}

// Convariance
pub fn covariance(x []f64, y []f64) f64 {
	assert x.len == y.len, "x and y should have the same number of elements"
	return linalg.dot(dev_mean(x), dev_mean(y)) / f64(x.len - 1)
}

// Correlation
pub fn correlation(x []f64, y []f64) f64 {
	stdev_x := standard_deviation(x)
	stdev_y := standard_deviation(y)
	return if stdev_x > 0 && stdev_y > 0 { covariance(x,y)/(stdev_x * stdev_y)} else { f64(0) }
}


