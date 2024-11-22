# VStats 0.0.1

A (very) basic Linear Algebra, Statistics, and Machine Learning library written from scratch and without dependencies

## Modules implemented

The following is the list of functions implemented so far

### Linear Algebra (linalg)

####  Vectors 

- `add(v []f64, w []f64) []f64`: Adds two vectors `a` and `b`: `(a + b)`
- `subtract(v []f64, w []f64) []f64`: Subtracts two vectors `a` and `b`: `(a - b)`
- `vector_sum(vector_list [][]f64) []f64`: Sums a list of vectors, example: `vector_sum([[f64(1),2],[3,4]]) => [4.0, 6.0]`
- `scalar_multiply(c f64, v []f64) []f64`: Multiplies an scalar value `c` to each element of a vector `v`
- `vector_mean(vector_list [][]f64) []f64`: Calculates 1/n sum_j (v[j])
- `dot(v []f64, w []f64) f64`: Dot product of `v` and `w`
- `sum_of_squares(v []f64) f64`: Squares each term of a vector, example: [1,2,3]^2 = [1^2, 2^2, 3^2]
- `magnitude(v []f64) f64`: Module of a vector, example: || [3,4] || = 5
- `squared_distance(v []f64, w []f64) f64`: Calculates sqrt[(v1-w1)^2 + (v2-w2)^2...]
- `distance(v []f64, w []f64) f64`: Calculates the distance between `v` and `w`

#### Matrices

- `shape(a [][]f64) (int, int)`: Returns the shape of a matrix (rows, columns)
- `get_row(a [][]f64, i int) []f64`: Gets the i-th row of a matrix as a vector
- `get_column(a [][]f64, j int) []f64`: Gets the j-th column of a matrix as a vector
- `make_matrix(num_rows int, num_cols int, op fn (int, int) f64) [][]f64`: Makes a matrix using a formula given by function `op`
- `identity_matrix(n int) [][]f64`: Returns a n-identity matrix
- `matmul(a [][]f64, b [][]f64) [][]f64`: Multuplies matrix `a` with `b`

### Probabilites

#### Distributions (CDF and PDF)

- `beta_function(x f64, y f64) f64`
- `normal_cdf(x f64, mu f64, sigma f64) f64`
- `inverse_normal_cdf(p f64, mu f64, sigma f64, dp DistribParams) f64`
- `bernoulli_pdf(x f64, p f64) f64`
- `bernoulli_cdf(x f64, p f64) f64`
- `binomial_pdf(k int, n int, p f64) f64`
- `poisson_pdf(k int, lambda f64) f64`
- `poisson_cdf(k int, lambda f64) f64`
- `exponential_pdf(x f64, lambda f64) f64`
- `exponential_cdf(x f64, lambda f64) f64`
- `gamma_pdf(x f64, k f64, theta f64) f64`
- `chi_squared_pdf(x f64, df int) f64`
- `students_t_pdf(x f64, df int) f64`
- `f_distribution_pdf(x f64, d1 int, d2 int) f64`
- `beta_pdf(x f64, alpha f64, beta f64) f64`
- `uniform_pdf(x f64, a f64, b f64) f64`
- `uniform_cdf(x f64, a f64, b f64) f64`
- `negative_binomial_pdf(k int, r int, p f64) f64`
- `negative_binomial_cdf(k int, r int, p f64) f64`
- `multinomial_pdf(x []int, p []f64) f64`
- `expectation(x []f64, p []f64) f64`

### Statistics

- `sum(x []f64) f64`
- `mean(x []f64) f64`
- `median(x []f64) f64`
- `quantile(x []f64, p f64) f64`
- `mode(x []f64) []f64`
- `data_range(x []f64) f64`
- `dev_mean(x []f64) []f64`
- `variance(x []f64) f64`
- `standard_deviation(x []f64) f64`
- `interquartile_range(x []f64) f64`
- `covariance(x []f64, y []f64) f64`
- `correlation(x []f64, y []f64) f64`

### Optimization

- `difference_quotient(f fn (f64) f64, x f64, h f64) f64`
- `partial_difference_quotient(f fn([]f64) f64, v []f64, i int, h f64) f64`
- `gradient(f fn([]f64) f64, v []f64, h f64) []f64`
- `gradient_step(v []f64, gradient_vector []f64, step_size f64) []f64`
- `sum_of_squares_gradient(v []f64) []f64`

## Disclaimer

- This was written as an exercise to get V closer to Data Analytics and Machine Learning tasks
- Heavily inspired by the book from Joel Grus "Data Science from Scratch: First principles with Python"
- It is **not** optimized in any way (at least for now)
- Documentation es an ongoing effort

## Roadmap

- Add more optimization algorithms
- Complete Hypothesis testing module 
- Complete Machine Learning module
- Complete Neural Network module
- Symbolic calculation

**Pull requests are welcome!**