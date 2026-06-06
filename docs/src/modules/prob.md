# prob

`import vstats.prob`

Probability density functions, cumulative distribution functions, and inverse CDFs
for common distributions.

> **vs Python:** mirrors `scipy.stats` distribution methods.
> `normal_pdf(x, mu, sigma)` corresponds to `scipy.stats.norm.pdf(x, loc=mu, scale=sigma)`.

## Normal

```v
normal_pdf(x f64, mu f64, sigma f64) f64
normal_cdf(x f64, mu f64, sigma f64) f64
inverse_normal_cdf(p f64, mu f64, sigma f64, tolerance f64) f64
```

## Discrete

```v
binomial_pmf(n int, k int, p f64) f64
binomial_cdf(n int, k int, p f64) f64
poisson_pmf(k int, lambda f64) f64
poisson_cdf(k int, lambda f64) f64
```

## Continuous

```v
beta_pdf(x f64, alpha f64, beta f64) f64
beta_cdf(x f64, alpha f64, beta f64) f64
gamma_pdf(x f64, shape f64, rate f64) f64
gamma_cdf(x f64, shape f64, rate f64) f64
exponential_pdf(x f64, lambda f64) f64
exponential_cdf(x f64, lambda f64) f64
uniform_pdf(x f64, a f64, b f64) f64
uniform_cdf(x f64, a f64, b f64) f64
t_pdf(x f64, df f64) f64
t_cdf(x f64, df f64) f64
chi_squared_pdf(x f64, k f64) f64
chi_squared_cdf(x f64, k f64) f64
f_pdf(x f64, d1 f64, d2 f64) f64
f_cdf(x f64, d1 f64, d2 f64) f64
```
