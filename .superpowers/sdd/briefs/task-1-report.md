# Task 1 Report: Audit `experiment/abtest.v`

## Status
DONE — no bugs found, no changes made to source.

## Functions Audited

| Function | Verdict |
|---|---|
| `welch_t` | Correct |
| `t_pvalue` | Correct |
| `t_pvalue_approx` | Correct |
| `welch_ci` | Correct |
| `abtest` (two-sided) | Correct |
| `abtest` (.greater one-sided) | Correct |
| `abtest` (.less one-sided) | Correct |
| `power_analysis` | Correct |
| `proportion_power_analysis` | Correct |
| `ancova` | Correct |
| `cuped_test` | Correct |
| `itt_and_pp` | Correct |

## Detailed Findings

### `welch_t`
- Welch-Satterthwaite df: `num = (a+b)²`, `den = a²/(n1-1) + b²/(n2-1)` — matches canonical formula exactly.
- SE: `sqrt(v1/n1 + v2/n2)` — correct.
- t: `(m1-m2)/se` — correct.

### `t_pvalue`
- Two-sided: `2 * CDF(-|t|, df)` — correct.

### `abtest` one-sided p-values
- `.greater` (trt > ctrl): t = (m_t - m_c)/se, p = `students_t_cdf(-t, df)` = P(T > t) — correct.
- `.less` (trt < ctrl): p = `students_t_cdf(t, df)` = P(T < t) — correct.
- CI always two-sided regardless of alternative — intentional by design.

### `power_analysis`
- Formula: `n = 2 * ((z_α/2 + z_β) / d)²` — matches exactly.
- Ceiling applied correctly via `if n_raw > f64(int(n_raw)) { int(n_raw) + 1 } else { int(n_raw) }`.

### `proportion_power_analysis`
- Fleiss formula: `n = (z_α * sqrt(2*p̄*(1-p̄)) + z_β * sqrt(p₁*(1-p₁) + p₂*(1-p₂)))² / (p₁-p₂)²`
- `p_bar = (p_baseline + p_treatment) / 2` — correct for the Fleiss formulation.
- Spot-check (alpha=0.05, power=0.80, p_baseline=0.10, p_treatment=0.15): test asserts `|n - 686| <= 5`, passes. The brief's target range was 3600-4000 for p_treatment=0.12; the test uses p_treatment=0.15 which naturally gives ~686 — not contradicted.

### `ancova`
- OLS with treatment indicator `x_design[i][0]` = 0 or 1, covariates at indices 1..n_cov — correct.
- `model.coefficients[0]` = treatment effect (adjusted) — correct.
- `full_coefs = [intercept] + coefficients` — `se_vec[1]` = SE of treatment — correct indexing.
- DF: `n - (n_cov + 2)` — correct (intercept + treatment + n_cov params).

### `cuped_test`
- θ = `cov(Y, X_pre) / var(X_pre)` over pooled ctrl+trt — correct.
- `Y_adj = Y - θ*(X_pre - mean_pre)` — correct.
- `variance_reduction = θ² * var(X_pre) / var(Y)` — correct.

### `itt_and_pp`
- ITT: splits by `assigned[i]`, ignores `complied` — correct.
- PP: skips `!complied[i]`, then splits by `assigned[i]` — both arms filtered — correct.

## Bugs Confirmed and Fixed
None.

## Uncertain Findings (flagged, not changed)
None.

## Test Results
Command: `v test tests/`
Result: **42 passed, 42 total** — no regressions.

## Commit
No source changes made. Audit commit: this report file only.
