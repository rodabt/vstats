module utils

import stats

// Helper functions
pub fn factorial(n int) f64 {
    if n <= 0 {
        return 1.0
    }
    mut result := 1.0
    for i := 1; i <= n; i++ {
        result *= i
    }
    return result
}

pub fn combinations(n int, k int) f64 {
    return factorial(n) / (factorial(k) * factorial(n - k))
}

pub fn permutations(n int, k int) f64 {
    return factorial(n) / factorial(n - k)
}

pub fn ipow(base int, exp int) int {
    mut result := 1
    for _ in 0..exp {
        result *= base
    }
    return result
}

pub fn range(n int) []int {
    mut r := []int{}
    for i in 0..n {
        r << i
    }
    return r
}

// Feature normalization: standardize features using training set statistics
// Returns (normalized_data, feature_means, feature_stds)
pub fn normalize_features(x [][]f64) ([][]f64, []f64, []f64) {
    assert x.len > 0, "Cannot normalize empty dataset"
    
    n_features := x[0].len
    mut means := []f64{len: n_features}
    mut stds := []f64{len: n_features}
    
    // Calculate mean and std for each feature
    for j in 0..n_features {
        feature_col := x.map(it[j])
        means[j] = stats.mean(feature_col)
        stds[j] = stats.standard_deviation(feature_col)
    }
    
    // Normalize data using computed statistics
    mut x_norm := [][]f64{}
    for i in 0..x.len {
        mut row := []f64{}
        for j in 0..n_features {
            std_safe := if stds[j] > 0 { stds[j] } else { 1.0 }
            row << (x[i][j] - means[j]) / std_safe
        }
        x_norm << row
    }
    
    return x_norm, means, stds
}

// Apply pre-computed normalization to new data
pub fn apply_normalization(x [][]f64, means []f64, stds []f64) [][]f64 {
    assert x.len > 0, "Cannot normalize empty dataset"
    assert means.len == x[0].len, "means length must match feature count"
    assert stds.len == x[0].len, "stds length must match feature count"
    
    mut x_norm := [][]f64{}
    for i in 0..x.len {
        mut row := []f64{}
        for j in 0..x[i].len {
            std_safe := if stds[j] > 0 { stds[j] } else { 1.0 }
            row << (x[i][j] - means[j]) / std_safe
        }
        x_norm << row
    }
    
    return x_norm
}