# Time Series Module Design

**Date:** 2026-06-15
**Status:** Approved

## Goal

Add a `timeseries/` module to VStats covering classical time series analysis, univariate forecasting, and multivariate vector autoregression — closing the largest gap relative to Python's statsmodels.

## Module Structure

```
timeseries/
├── analysis.v      — ACF, PACF, unit root tests, differencing, information criteria
├── decomposition.v — Classical decomposition, STL
├── arima.v         — ARIMA, SARIMA, auto_arima, forecasting
├── smoothing.v     — Simple ES, Holt, Holt-Winters, auto-optimization
└── var.v           — VAR(p), lag selection, Granger causality, IRF
```

**Dependencies:** `vstats.stats`, `vstats.prob`, `vstats.linalg`. No new external dependencies.

**Layering:** `analysis.v` is the shared foundation consumed by `arima.v` and `var.v`. `decomposition.v` is standalone. `arima.v` and `smoothing.v` are parallel univariate forecasters. `var.v` is the multivariate layer.

---

## `analysis.v` — Foundation Tools

### Differencing

```v
fn diff(x []f64, d int) []f64
fn seasonal_diff(x []f64, period int) []f64
fn undiff(diffed []f64, original []f64, d int) []f64
```

`undiff` inverts d-order differencing to recover forecasts in the original scale.

### ACF / PACF

```v
fn acf(x []f64, nlags int) []f64
fn pacf(x []f64, nlags int) []f64
fn acf_confidence_bound(n int) f64  // Bartlett's ±1.96/√n
```

`pacf` uses Yule-Walker equations. `acf_confidence_bound` returns the threshold for significance at each lag.

### Unit Root Tests

```v
struct ADFResult  { statistic f64, p_value f64, is_stationary bool, lags_used int }
struct KPSSResult { statistic f64, p_value f64, is_stationary bool, lags_used int }

fn adf_test(x []f64, lags int) ADFResult
fn kpss_test(x []f64, lags int) KPSSResult
```

ADF null hypothesis: unit root (non-stationary). KPSS null hypothesis: stationary (opposite). ADF p-values use MacKinnon's response surface approximation. KPSS uses tabulated critical values at standard significance levels.

### Information Criteria

```v
fn aic(log_likelihood f64, k int) f64
fn bic(log_likelihood f64, k int, n int) f64
fn aicc(log_likelihood f64, k int, n int) f64
```

Shared by `auto_arima` and `var_select_lag`.

---

## `decomposition.v` — Decomposition

### Classical Decomposition

```v
enum DecompositionModel { additive, multiplicative }

struct ClassicalDecomposition {
    trend    []f64
    seasonal []f64
    residual []f64
    model    DecompositionModel
}

fn decompose(x []f64, period int, model DecompositionModel) ClassicalDecomposition
```

Centered moving average for trend; period averages for seasonal; residual by subtraction (additive) or division (multiplicative).

### STL Decomposition

```v
struct STLConfig {
    seasonal_window int
    trend_window    int
    n_iter          int  // default 2
}

struct STLResult {
    trend    []f64
    seasonal []f64
    residual []f64
}

fn stl(x []f64, period int, cfg STLConfig) STLResult
```

Iterative LOESS smoothing: inner loop fits seasonal then trend components, outer loop applies robustness weights to downweight anomalies. More robust than classical decomposition for real-world data. LOESS smoother is a private function not exported from the module.

---

## `arima.v` — ARIMA & SARIMA

### Structs

```v
struct ARIMAModel {
    p         int
    d         int
    q         int
    ar_coeffs []f64    // AR parameters φ₁..φₚ
    ma_coeffs []f64    // MA parameters θ₁..θq
    intercept f64
    sigma2    f64      // residual variance
    fitted    []f64
    residuals []f64
    aic       f64
    bic       f64
    aicc      f64
}

struct ForecastResult {
    forecast []f64
    lower    []f64    // CI lower bound
    upper    []f64    // CI upper bound
    alpha    f64      // CI level (default 0.05 → 95% CI)
}
```

### Functions

```v
fn arima_fit(x []f64, p int, d int, q int) ARIMAModel
fn sarima_fit(x []f64, p int, d int, q int, P int, D int, Q int, m int) ARIMAModel
fn arima_forecast(model ARIMAModel, steps int, alpha f64) ForecastResult
fn arima_summary(model ARIMAModel) string
fn auto_arima(x []f64, max_p int, max_q int, max_d int) ARIMAModel
```

**Estimation:** Conditional Sum of Squares (CSS). Numerically stable, avoids Kalman filter complexity, produces results equivalent to full MLE for moderate-length series.

**SARIMA:** Applies seasonal differencing (order D, period m), then fits the combined seasonal+non-seasonal AR/MA polynomials.

**`arima_summary`:** Formatted table of coefficients, standard errors, t-statistics, and information criteria — mirrors statsmodels' `.summary()` style.

**`auto_arima`:** Grid search over all (p,d,q) combinations up to given maxima; selects by AICc. Runs ADF test first to determine the minimum d needed for stationarity; `max_d` acts as an upper bound cap on that result.

**CIs:** Analytical formula for AR(p) propagating variance through the MA(∞) representation.

---

## `smoothing.v` — Exponential Smoothing

### Structs

```v
enum HWSeasonalType { additive, multiplicative }

struct SmoothingResult {
    fitted   []f64
    forecast []f64    // h-step ahead point forecasts
    level    []f64
    trend    []f64    // zero-length if unused (SES)
    seasonal []f64    // zero-length if unused (SES/Holt)
    alpha    f64
    beta     f64      // 0.0 if unused
    gamma    f64      // 0.0 if unused
    mse      f64
}
```

### Functions

```v
fn ses(x []f64, alpha f64) SmoothingResult
fn holt(x []f64, alpha f64, beta f64) SmoothingResult
fn holt_winters(x []f64, alpha f64, beta f64, gamma f64, period int, seasonal HWSeasonalType) SmoothingResult

fn auto_ses(x []f64) SmoothingResult
fn auto_holt(x []f64) SmoothingResult
fn auto_holt_winters(x []f64, period int, seasonal HWSeasonalType) SmoothingResult
```

**SES:** Level-only; `lₜ = α·xₜ + (1−α)·lₜ₋₁`. For series with no trend or seasonality.

**Holt:** Adds linear trend component. For trended, non-seasonal series.

**Holt-Winters:** Adds seasonal component. Additive when seasonal amplitude is constant; multiplicative when amplitude scales with level. Initial seasonal indices use classical period-averaging.

**Auto-optimization:** Nelder-Mead simplex search over MSE. Implemented privately within the file (bounded 2-3 parameter search; does not require `optim/` integration).

---

## `var.v` — Vector Autoregression

### Structs

```v
struct VARModel {
    p               int
    k               int           // number of variables
    coeff_matrices  [][][]f64     // k × (k*p + 1) per equation
    sigma_u         [][]f64       // residual covariance matrix
    fitted          [][]f64
    residuals       [][]f64
    aic             f64
    bic             f64
    hqc             f64           // Hannan-Quinn criterion
}

struct VARLagSelection {
    aic_order int
    bic_order int
    hqc_order int
    criteria  [][]f64             // full table: rows=lags, cols=[AIC, BIC, HQC]
}

struct GrangerResult {
    cause_var   int
    effect_var  int
    f_statistic f64
    p_value     f64
    significant bool
}

struct IRFResult {
    responses [][][]f64           // [shock_var][response_var][period]
    periods   int
}
```

### Functions

```v
fn var_fit(data [][]f64, p int) VARModel
fn var_forecast(model VARModel, steps int) [][]f64  // shape: [k][steps]
fn var_select_lag(data [][]f64, max_p int) VARLagSelection
fn granger_causality(data [][]f64, p int, cause_idx int, effect_idx int) GrangerResult
fn irf(model VARModel, steps int) IRFResult
```

**Estimation:** OLS equation-by-equation. Each variable is regressed on all variables' lags. Equivalent to GLS when the same regressors appear in every equation (the VAR case). Reuses `linalg` normal equations.

**`var_select_lag`:** Fits VAR(1)..VAR(max_p), computes AIC/BIC/HQC at each order, returns recommended order per criterion plus the full criteria table.

**`granger_causality`:** F-test comparing restricted model (without cause variable's lags) vs unrestricted. Standard Granger definition.

**`irf`:** Traces the effect of a one-unit shock to each variable on all others over `steps` periods. Uses Cholesky orthogonalization of `sigma_u` for orthogonalized IRFs.

---

## Test Coverage

All tests live in `tests/` per VStats convention.

| File | Test cases |
|------|-----------|
| `timeseries_analysis_test.v` | ACF/PACF on known AR(1); ADF rejects unit root on stationary series; KPSS accepts stationarity; diff/undiff roundtrip |
| `timeseries_decomposition_test.v` | Classical additive/multiplicative recovers synthetic components; STL on seasonal series |
| `timeseries_arima_test.v` | AR(1) coefficient recovery; ARIMA(1,1,0) on differenced series; forecast shape and CI coverage |
| `timeseries_smoothing_test.v` | SES on constant series gives constant fit; Holt-Winters seasonal indices sum to zero (additive); auto_ses MSE ≤ naive |
| `timeseries_var_test.v` | VAR(1) coefficient recovery on known bivariate system; Granger causality detects known causal structure; IRF shape |

---

## Implementation Notes

- `pacf` via Yule-Walker requires solving a Toeplitz system — use Levinson-Durbin recursion (O(p²), numerically stable)
- ADF MacKinnon approximation uses response surface coefficients from MacKinnon (1994) — constants embedded as compile-time arrays
- CSS ARIMA initialization: backcast the pre-sample residuals to zero
- Nelder-Mead in `smoothing.v` operates on log-transformed parameters to enforce positivity constraints naturally
- VAR Cholesky for IRF: `linalg` has no Cholesky decomposition; implement a private `cholesky(m [][]f64) [][]f64` in `var.v`
