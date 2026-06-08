module chart

import math

pub struct HistogramBins {
pub:
	edges  []f64 // len == nbins + 1
	counts []int // len == nbins
}

pub fn histogram_bins(data []f64, nbins int) HistogramBins {
	assert data.len > 0
	n := if nbins > 0 {
		nbins
	} else {
		int(math.ceil(math.log2(f64(data.len)) + 1.0)) // Sturges' rule
	}
	mut lo := data[0]
	mut hi := data[0]
	for v in data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if lo == hi {
		hi = lo + 1.0
	}
	width := (hi - lo) / f64(n)
	mut edges := []f64{len: n + 1}
	for i in 0 .. n + 1 {
		edges[i] = lo + f64(i) * width
	}
	mut counts := []int{len: n, init: 0}
	for v in data {
		mut idx := int((v - lo) / width)
		if idx >= n {
			idx = n - 1
		}
		if idx < 0 {
			idx = 0
		}
		counts[idx]++
	}
	return HistogramBins{
		edges:  edges
		counts: counts
	}
}
