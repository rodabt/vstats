# VStats

A dependency-free statistics, linear algebra, and machine learning library for V.

---

**New here?** → [Getting Started (Python/R users)](getting-started.html)

**Know what you want?** → [Module Reference](modules/stats.html)

---

## What is VStats?

VStats gives you the statistical and ML toolkit you'd normally reach for in Python
(scipy, sklearn, statsmodels) — in pure V, with no external dependencies.

- **Zero dependencies** — ships as pure V source
- **Generic types** — most functions accept `int` or `f64`; aggregations return `f64`
- **Product analytics focus** — `experiment` and `growth` modules cover A/B testing,
  causal inference, funnels, attribution, and SaaS metrics out of the box

## Modules

| Module | Purpose |
|--------|---------|
| `linalg` | Vectors and matrices |
| `stats` | Descriptive stats, multiple testing, delta method, bootstrap |
| `prob` | Probability distributions (PDF/CDF/inverse) |
| `optim` | Gradient descent |
| `utils` | Datasets, metrics, feature scaling |
| `ml` | Regression, classification, clustering |
| `nn` | Neural network layers and training |
| `hypothesis` | Statistical tests |
| `experiment` | A/B testing, CUPED, DiD, PSM, ANCOVA, ITT/PP |
| `growth` | Funnels, cohorts, attribution, SaaS metrics |
| `chart` | SVG charts (line, scatter, bar, histogram) with Tufte defaults |

## Install

```bash
v install https://github.com/rodabt/vstats
```

## Build & Test

```bash
make test       # run all tests
make fulltest   # verbose output
make docs       # regenerate HTML docs from docs/src/
```
