module utils

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

pub fn range(n int) []int {
    mut r := []int{}
    for i in 0..n {
        r << i
    }
    return r
}