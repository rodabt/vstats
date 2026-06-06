# Cluster Sampling Calculators — Design Spec

**Date:** 2026-04-23  
**Status:** Approved

## Context

Cluster-randomized experiments (e.g., randomizing by account rather than user) violate the independence assumption behind standard sample size formulas. The intraclass correlation (ICC / ρ) quantifies within-cluster homogeneity; the design effect (DEFF) translates that into a sample size multiplier. Without these tools, experimenters routinely underpower cluster-randomized tests.

This adds two new independent calculators to the vstats web app.

---

## Calculator 1: ICC Estimation

**URL:** `/calculators/icc`  
**Nav label:** ICC Estimation

### When to use
You have clustered data (users in accounts, students in classrooms, stores in regions) and need to quantify how similar units within the same cluster are before designing a cluster-randomized experiment.

### Inputs
| Field | Type | Default |
|-------|------|---------|
| Data (`cluster_id, value` per row) | textarea | — |
| Alpha | number | 0.05 |

### Outputs (stat-grid)
| Metric | Description |
|--------|-------------|
| ICC (ρ) | Intraclass correlation coefficient |
| Clusters (k) | Number of distinct clusters parsed |
| Avg cluster size (m̄) | Mean units per cluster |
| F-statistic | From one-way ANOVA |
| p-value | Tests H₀: ρ = 0 |

### Interpretation
"ρ = X means X% of outcome variance is explained by cluster membership. Typical values in digital experiments range from 0.01–0.30."

### Library implementation
New file: `experiment/cluster.v`

```
fn icc_estimate(groups [][]f64) (f64, f64, f64)
  // Returns (rho, F_statistic, p_value)
  // Calls existing anova_one_way() from stats/descriptive.v for MSB, MSW
  // Applies unbalanced-design correction:
  //   m0 = (1/(k-1)) * (N - sum(n_i^2)/N)
  //   rho = (MSB - MSW) / (MSB + (m0 - 1) * MSW)
```

### Backend
- New route: `@['/api/icc'; post]` in `web/api_icc.v`
- Request: `{ data: string, alpha: f64 }` — server parses CSV, groups by cluster_id
- Response: `{ rho, k, avg_cluster_size, f_statistic, p_value, significant }`

### Load example
~30 rows across 5 clusters, moderate ICC (~0.15) so results are meaningful.

---

## Calculator 2: DEFF Adjustment

**URL:** `/calculators/deff`  
**Nav label:** Design Effect

### When to use
You've computed a required n assuming simple random sampling (e.g., from the Power Analysis calculator), but your experiment randomizes by cluster. Apply the design effect to get the true required sample size.

### Inputs
| Field | Type | Default |
|-------|------|---------|
| ICC (ρ) | number | — |
| Average cluster size (m̄) | number | — |
| Simple random sample n | number | — |

### Outputs (stat-grid)
| Metric | Formula |
|--------|---------|
| DEFF | 1 + (m̄ − 1) × ρ |
| Adjusted n | ⌈n × DEFF⌉ |
| Clusters needed | ⌈adjusted n / m̄⌉ |

### Interpretation
"With ρ = X and cluster size m̄, DEFF = Y — you need Y× more subjects than a simple random sample requires."

### Library implementation
In `experiment/cluster.v`:

```
fn design_effect(rho f64, m f64) f64
  // returns 1.0 + (m - 1.0) * rho

fn adjusted_sample_size(n int, deff f64) int
  // returns ceil(n * deff)
```

### Backend
- New route: `@['/api/deff'; post]` in `web/api_deff.v`
- Request: `{ rho: f64, m: f64, n: int }`
- Response: `{ deff, adjusted_n, clusters_needed }`

### Load example
ρ = 0.12, m̄ = 50, n = 1000 → DEFF = 6.88, adjusted n = 6880, clusters = 138.

---

## Files Changed

| File | Change |
|------|--------|
| `experiment/cluster.v` | New — `icc_estimate`, `design_effect`, `adjusted_sample_size` |
| `web/api_icc.v` | New — `/api/icc` POST endpoint |
| `web/api_deff.v` | New — `/api/deff` POST endpoint |
| `web/templates/icc.html` | New — ICC estimation page |
| `web/templates/deff.html` | New — DEFF adjustment page |
| `web/pages.v` | Add 2 page routes |
| `web/templates/_header.html` | Add 2 nav links |
| `web/static/js/calculators.js` | Add `iccCalc()` and `deffCalc()` Alpine components |
| `tests/cluster_test.v` | New — unit tests for `icc_estimate`, `design_effect`, `adjusted_sample_size` |

---

## Reused Patterns
- `anova_one_way()` from `stats/descriptive.v` — provides MSB, MSW
- `apiFetch()` from `web/static/js/utils.js`
- CSS classes: `.card`, `.result-card`, `.stat-grid`, `.stat-box`, `.did-explainer`, `.field-row`, `.error-banner`, `.spinner`
- `api_error()` from `web/helpers.v`

---

## Verification
1. `v test tests/` — all existing tests pass
2. Add `tests/cluster_test.v` — test `icc_estimate` on balanced clusters (known ρ), test `design_effect` on standard values (ρ=0, ρ=1, typical ρ=0.1)
3. Start server: `v run web/main.v`
4. Load ICC page → paste example → verify ρ, k, m̄, F, p match hand-computed values
5. Load DEFF page → enter ρ=0.12, m=50, n=1000 → verify DEFF=6.88, adjusted n=6880, clusters=138
6. Verify nav links appear and are marked active on each page
