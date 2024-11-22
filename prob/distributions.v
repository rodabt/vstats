module prob

import math
import linalg
import utils


@[params]
pub struct DistribParams {
	tolerance	f64 = 0.00001
}

pub fn beta_function(x f64, y f64) f64 {
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)
}

// Normal CDF
pub fn normal_cdf(x f64, mu f64, sigma f64) f64 {
	assert sigma > f64(0), "sigma should always be positive"
	return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
}

// Normal PDF
pub fn inverse_normal_cdf(p f64, mu f64, sigma f64, dp DistribParams) f64 {
	assert sigma > f64(0), "sigma should always be positive"
	if mu != f64(0) || sigma != f64(1) {
		return mu + sigma * inverse_normal_cdf(p, 0, 1)
	}
	mut low_z := f64(-10.0)
	mut hi_z := f64(10.0)
	mut mid_z := f64(0.0)
	for hi_z - low_z > dp.tolerance {
		mid_z = (low_z + hi_z) / 2.0
		mid_p := normal_cdf(mid_z, 0, 1)
		if mid_p < p {
			low_z = mid_z
		} else {
			hi_z = mid_z
		}
	}
	return mid_z
}


// Bernoulli PDF
pub fn bernoulli_pdf(x f64, p f64) f64 {
    assert x == 0 || x == 1, "x shoud only be 0 or 1"
    return if x == 1 { p } else { 1.0 - p }
}

// Bernoulli CDF
pub fn bernoulli_cdf(x f64, p f64) f64 {
    if x < 0 {
        return 0.0
    }
    if x >= 1 {
        return 1.0
    }
    return 1.0 - p
}


// Binomial PDF
pub fn binomial_pdf(k int, n int, p f64) f64 {
    assert k>=0 && k<=n, "k should be between 0 and n"
    coef := utils.combinations(n, k)
    return coef * math.pow(p, k) * math.pow(1.0 - p, n - k)
}

// Poisson PDF
pub fn poisson_pdf(k int, lambda f64) f64 {
	assert k>=0, "k should be non-negative"
    return math.exp(-lambda) * math.pow(lambda, k) / utils.factorial(k)
}

// Poisson CDF
pub fn poisson_cdf(k int, lambda f64) f64 {
    if k < 0 {
        return 0.0
    }
    mut sum := 0.0
    for i := 0; i <= k; i++ {
        sum += poisson_pdf(i, lambda)
    }
    return sum
}


// Exponential PDF
pub fn exponential_pdf(x f64, lambda f64) f64 {
    if x < 0 {
        return 0.0
    }
    return lambda * math.exp(-lambda * x)
}

// Exponential CDF
pub fn exponential_cdf(x f64, lambda f64) f64 {
    if x < 0 {
        return 0.0
    }
    return 1.0 - math.exp(-lambda * x)
}


// Gamma PDF
pub fn gamma_pdf(x f64, k f64, theta f64) f64 {
    if x < 0 {
        return 0.0
    }
    return math.pow(x, k-1) * math.exp(-x/theta) / (math.gamma(k) * math.pow(theta, k))
}

// Chi-squared PDF
pub fn chi_squared_pdf(x f64, df int) f64 {
    return gamma_pdf(x, f64(df)/2.0, 2.0)
}


// t-Student's PDF
pub fn students_t_pdf(x f64, df int) f64 {
    coef := math.gamma((f64(df) + 1.0) / 2.0) / (math.sqrt(f64(df) * math.pi) * math.gamma(f64(df) / 2.0))
    return coef * math.pow(1.0 + (x * x) / df, -(f64(df) + 1.0) / 2.0)
}


// F-Distribution PDF
pub fn f_distribution_pdf(x f64, d1 int, d2 int) f64 {
    if x < 0 {
        return 0.0
    }
    mut coef := math.sqrt(math.pow(d1 * x, d1) * math.pow(d2, d2) / math.pow(d1 * x + d2, d1 + d2))
    coef *= math.gamma((f64(d1) + f64(d2)) / 2.0) / (math.gamma(f64(d1)/2.0) * math.gamma(f64(d2)/2.0))
    return coef / x
}


// Beta PDF
pub fn beta_pdf(x f64, alpha f64, beta f64) f64 {
    if x < 0 || x > 1 {
        return 0.0
    }
    return math.pow(x, alpha-1) * math.pow(1-x, beta-1) / beta_function(alpha, beta)
}


// Uniform PDF
pub fn uniform_pdf(x f64, a f64, b f64) f64 {
    if x < a || x > b {
        return 0.0
    }
    return 1.0 / (b - a)
}

// Uniform CDF
pub fn uniform_cdf(x f64, a f64, b f64) f64 {
    if x < a {
        return 0.0
    }
    if x > b {
        return 1.0
    }
    return (x - a) / (b - a)
}


// Negative Binomial PDF
pub fn negative_binomial_pdf(k int, r int, p f64) f64 {
    if k < r {
        return 0.0
    }
    coef := utils.combinations(k - 1, r - 1)
    return coef * math.pow(p, r) * math.pow(1.0 - p, k - r)
}

// Negative Binomoal PDF
pub fn negative_binomial_cdf(k int, r int, p f64) f64 {
    if k < r {
        return 0.0
    }
    mut sum := 0.0
    for i := r; i <= k; i++ {
        sum += negative_binomial_pdf(i, r, p)
    }
    return sum
}


// Multinomial Distribution PDF
pub fn multinomial_pdf(x []int, p []f64) f64 {
    if x.len != p.len {
        return 0.0
    }
    
    // Check if probabilities sum to 1
    mut p_sum := 0.0
    for prob in p {
        p_sum += prob
    }
    if math.abs(p_sum - 1.0) > 1e-10 {
        return 0.0
    }
    
    // Calculate n (total number of trials)
    mut n := 0
    for count in x {
        if count < 0 {
            return 0.0
        }
        n += count
    }
    
    // Calculate multinomial coefficient
    mut coef := utils.factorial(n)
    for count in x {
        coef /= utils.factorial(count)
    }
    
    // Calculate product of probabilities
    mut prob_product := 1.0
    for i := 0; i < x.len; i++ {
        prob_product *= math.pow(p[i], x[i])
    }
    
    return coef * prob_product
}

// Discrete random var expectation
pub fn expectation(x []f64, p []f64) f64 {
	return linalg.dot(x, p)
}