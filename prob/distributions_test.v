module prob

import math

fn test__beta_function() {
	// beta(2, 2) = 1/6
	result := beta_function(2.0, 2.0)
	assert result > 0.166 && result < 0.167, "beta(2,2) should be ~0.1667"
}

fn test__normal_cdf() {
	// Normal CDF at mu should be 0.5
	assert math.abs(normal_cdf(0.0, 0.0, 1.0) - 0.5) < 0.001
	// CDF values increase with x
	assert normal_cdf(0.0, 0.0, 1.0) < normal_cdf(1.0, 0.0, 1.0)
}

fn test__bernoulli_pdf() {
	assert bernoulli_pdf(1.0, 0.5) == 0.5
	assert bernoulli_pdf(0.0, 0.5) == 0.5
	assert bernoulli_pdf(1.0, 0.7) == 0.7
	assert bernoulli_pdf(0.0, 0.3) == 0.7
}

fn test__bernoulli_cdf() {
	assert bernoulli_cdf(-1.0, 0.5) == 0.0
	assert bernoulli_cdf(0.5, 0.5) == 0.5
	assert bernoulli_cdf(1.0, 0.5) == 1.0
}

fn test__binomial_pdf() {
	// Binomial(n=5, k=2, p=0.5) should be C(5,2) * 0.5^2 * 0.5^3 = 10 * 0.03125 = 0.3125
	result := binomial_pdf(2, 5, 0.5)
	assert math.abs(result - 0.3125) < 0.001
}

fn test__poisson_pdf() {
	// Poisson(lambda=2, k=0) = e^(-2) = 0.1353...
	result := poisson_pdf(0, 2.0)
	assert result > 0.135 && result < 0.136, "Poisson(2,0) should be ~0.1353"
}

fn test__poisson_cdf() {
	// Poisson CDF is cumulative, should be >= PDF
	assert poisson_cdf(0, 2.0) <= poisson_cdf(1, 2.0)
	assert poisson_cdf(1, 2.0) <= poisson_cdf(2, 2.0)
}

fn test__exponential_pdf() {
	assert exponential_pdf(-1.0, 1.0) == 0.0
	assert exponential_pdf(0.0, 1.0) == 1.0
	// Exponential PDF is monotonic decreasing
	assert exponential_pdf(0.5, 1.0) > exponential_pdf(1.0, 1.0)
}

fn test__exponential_cdf() {
	assert exponential_cdf(-1.0, 1.0) == 0.0
	assert exponential_cdf(0.0, 1.0) == 0.0
	// CDF should be increasing
	assert exponential_cdf(0.5, 1.0) < exponential_cdf(1.0, 1.0)
}

fn test__uniform_pdf() {
	// Uniform on [0,1] should be 1.0 everywhere in range
	assert uniform_pdf(0.5, 0.0, 1.0) == 1.0
	assert uniform_pdf(-0.5, 0.0, 1.0) == 0.0
	assert uniform_pdf(1.5, 0.0, 1.0) == 0.0
}

fn test__uniform_cdf() {
	assert uniform_cdf(-1.0, 0.0, 1.0) == 0.0
	assert uniform_cdf(0.5, 0.0, 1.0) == 0.5
	assert uniform_cdf(2.0, 0.0, 1.0) == 1.0
}

fn test__negative_binomial_pdf() {
	// If k < r, should be 0
	assert negative_binomial_pdf(2, 5, 0.5) == 0.0
	// r=1 should give geometric distribution
	result := negative_binomial_pdf(3, 1, 0.5)
	assert result > 0.0
}

fn test__negative_binomial_cdf() {
	assert negative_binomial_cdf(3, 5, 0.5) == 0.0
	// CDF should be non-decreasing
	r, p := 2, 0.5
	cdf1 := negative_binomial_cdf(5, r, p)
	cdf2 := negative_binomial_cdf(6, r, p)
	assert cdf1 <= cdf2
}

fn test__multinomial_pdf() {
	// Valid multinomial
	result := multinomial_pdf([1, 1, 1], [0.33, 0.33, 0.34])
	assert result > 0.0
	// Mismatched lengths
	assert multinomial_pdf([1, 1], [0.5, 0.5, 0.0]) == 0.0
	// Probabilities don't sum to 1
	assert multinomial_pdf([1, 1], [0.5, 0.6]) == 0.0
}

fn test__expectation() {
	// E[x] = sum(x[i] * p[i])
	x := [1.0, 2.0, 3.0]
	p := [0.2, 0.3, 0.5]
	result := expectation(x, p)
	expected := 1.0*0.2 + 2.0*0.3 + 3.0*0.5
	assert math.abs(result - expected) < 0.001
}
