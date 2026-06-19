module experiment

import math

// PVal is an internal helper for sorting p-values while tracking original indices.
struct PVal {
	original_idx int
	val          f64
}

pub struct CorrectionResult {
pub:
	method            string
	alpha             f64
	adjusted_p_values []f64
	rejected          []bool
	n_rejected        int
}

// bonferroni applies Bonferroni correction: rejects H_i if p_i < alpha/k.
// Controls the familywise error rate (FWER). Most conservative of the three methods.
pub fn bonferroni(p_values []f64, alpha f64) CorrectionResult {
	assert p_values.len > 0, 'p_values must not be empty'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	k := p_values.len
	mut adjusted := []f64{len: k}
	mut rejected := []bool{len: k}
	mut n_rej := 0
	for i, p in p_values {
		adj := math.min(p * f64(k), 1.0)
		adjusted[i] = adj
		rejected[i] = p < alpha / f64(k)
		if rejected[i] {
			n_rej++
		}
	}
	return CorrectionResult{
		method:            'bonferroni'
		alpha:             alpha
		adjusted_p_values: adjusted
		rejected:          rejected
		n_rejected:        n_rej
	}
}

// holm applies Holm's step-down correction.
// More powerful than Bonferroni while still controlling FWER.
pub fn holm(p_values []f64, alpha f64) CorrectionResult {
	assert p_values.len > 0, 'p_values must not be empty'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	k := p_values.len
	mut pairs := []PVal{len: k}
	for i in 0 .. k {
		pairs[i] = PVal{original_idx: i, val: p_values[i]}
	}
	pairs.sort(a.val < b.val)

	mut rejected := []bool{len: k}
	mut adjusted := []f64{len: k}
	mut stop := false
	mut running_max := 0.0
	for rank in 0 .. k {
		idx := pairs[rank].original_idx
		threshold := alpha / f64(k - rank)
		if !stop && p_values[idx] <= threshold {
			rejected[idx] = true
		} else {
			stop = true
		}
		// Adjusted p-value: enforce monotone increase along sorted order
		raw := math.min(f64(k - rank) * p_values[idx], 1.0)
		if raw > running_max {
			running_max = raw
		}
		adjusted[idx] = running_max
	}

	mut n_rej := 0
	for r in rejected {
		if r {
			n_rej++
		}
	}
	return CorrectionResult{
		method:            'holm'
		alpha:             alpha
		adjusted_p_values: adjusted
		rejected:          rejected
		n_rejected:        n_rej
	}
}

// bh_fdr applies Benjamini-Hochberg correction.
// Controls the false discovery rate (FDR). Recommended for product experiments
// where a small fraction of false positives is acceptable in exchange for more power.
pub fn bh_fdr(p_values []f64, alpha f64) CorrectionResult {
	assert p_values.len > 0, 'p_values must not be empty'
	assert alpha > 0 && alpha < 1, 'alpha must be in (0, 1)'
	k := p_values.len
	mut pairs := []PVal{len: k}
	for i in 0 .. k {
		pairs[i] = PVal{original_idx: i, val: p_values[i]}
	}
	pairs.sort(a.val < b.val)

	// Find largest rank where p_(rank+1) <= (rank+1)/k * alpha
	mut max_rank := -1
	for rank in 0 .. k {
		idx := pairs[rank].original_idx
		threshold := f64(rank + 1) / f64(k) * alpha
		if p_values[idx] <= threshold {
			max_rank = rank
		}
	}

	mut rejected := []bool{len: k}
	if max_rank >= 0 {
		for rank in 0 .. max_rank + 1 {
			rejected[pairs[rank].original_idx] = true
		}
	}

	// BH adjusted p-value: running minimum from highest to lowest rank
	mut adjusted := []f64{len: k}
	mut running_min := 1.0
	mut rank := k - 1
	for rank >= 0 {
		idx := pairs[rank].original_idx
		raw := math.min(f64(k) / f64(rank + 1) * p_values[idx], 1.0)
		if raw < running_min {
			running_min = raw
		}
		adjusted[idx] = running_min
		rank--
	}

	mut n_rej := 0
	for r in rejected {
		if r {
			n_rej++
		}
	}
	return CorrectionResult{
		method:            'bh_fdr'
		alpha:             alpha
		adjusted_p_values: adjusted
		rejected:          rejected
		n_rejected:        n_rej
	}
}
