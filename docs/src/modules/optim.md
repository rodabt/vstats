# optim

`import vstats.optim`

Gradient descent and numerical differentiation. Typically used internally by `ml` —
call directly when implementing custom objectives.

> **vs Python:** replaces manual gradient descent loops or `scipy.optimize.minimize`.

## Gradient Estimation

```v
partial_difference_quotient(f fn([]f64) f64, v []f64, i int, h f64) f64
estimate_gradient(f fn([]f64) f64, v []f64, h f64) []f64
```

## Gradient Descent

```v
gradient_step(v []f64, gradient []f64, step_size f64) []f64
gradient_descent(f fn([]f64) f64, gradient_f fn([]f64) []f64, start []f64, step_size f64, iterations int) []f64
```
