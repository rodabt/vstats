module stats

pub struct BHResult {
pub:
	adjusted  []f64
	reject    []bool
	n_rejected int
}

pub struct BonferroniResult {
pub:
	adjusted  []f64
	reject    []bool
	n_rejected int
}

pub fn bh_correction(p_values []f64, alpha f64) BHResult {
	n := p_values.len
	assert n > 0, 'p_values must not be empty'
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0, 1)'

	// Sort indices by p-value ascending (insertion sort — no external deps)
	mut order := []int{len: n, init: index}
	for i := 1; i < n; i++ {
		key := order[i]
		mut j := i - 1
		for j >= 0 && p_values[order[j]] > p_values[key] {
			order[j + 1] = order[j]
			j--
		}
		order[j + 1] = key
	}

	// Adjusted p-values: p_adj = p * n / rank  (rank is 1-based)
	mut adj := []f64{len: n}
	for rank, orig_idx in order {
		adj[orig_idx] = p_values[orig_idx] * f64(n) / f64(rank + 1)
	}

	// Enforce monotonicity (step-down): scan sorted order right to left
	mut min_so_far := 1.0
	for r := n - 1; r >= 0; r-- {
		orig_idx := order[r]
		if adj[orig_idx] < min_so_far {
			min_so_far = adj[orig_idx]
		} else {
			adj[orig_idx] = min_so_far
		}
	}

	// Clamp to [0, 1] and build rejection flags
	mut reject := []bool{len: n}
	mut n_rejected := 0
	for i in 0 .. n {
		if adj[i] > 1.0 {
			adj[i] = 1.0
		}
		reject[i] = adj[i] <= alpha
		if reject[i] {
			n_rejected++
		}
	}
	return BHResult{ adjusted: adj, reject: reject, n_rejected: n_rejected }
}

pub fn bonferroni_correction(p_values []f64, alpha f64) BonferroniResult {
	n := p_values.len
	assert n > 0, 'p_values must not be empty'
	assert alpha > 0.0 && alpha < 1.0, 'alpha must be in (0, 1)'
	mut adj := []f64{len: n}
	mut reject := []bool{len: n}
	mut n_rejected := 0
	for i, p in p_values {
		a := p * f64(n)
		adj[i] = if a > 1.0 { 1.0 } else { a }
		reject[i] = adj[i] <= alpha
		if reject[i] {
			n_rejected++
		}
	}
	return BonferroniResult{ adjusted: adj, reject: reject, n_rejected: n_rejected }
}
