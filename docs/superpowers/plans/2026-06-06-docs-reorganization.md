# Docs Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-written HTML docs with a Markdown-source build pipeline, reorganize `examples/` into 6 named scenarios, and add a "For Python/R users" getting-started guide.

**Architecture:** `docs/src/**/*.md` files are the canonical source. `docs/build.py` renders them to `docs/**/*.html` by expanding `<!-- include: -->` tags (which embed `.v` files verbatim) and applying a shared HTML scaffold. The existing `css/` and `js/` are reused unchanged.

**Tech Stack:** Python 3, `markdown` library (already installed for companion), V (`v run`, `v build`)

---

## File Map

**Create:**
- `docs/build.py` — build script
- `docs/src/index.md`, `getting-started.md`, `concepts.md`, `examples.md`
- `docs/src/modules/stats.md`, `experiment.md`, `ml.md`, `hypothesis.md`, `growth.md`, `nn.md`, `prob.md`, `optim.md`, `linalg.md`, `utils.md`
- `examples/rigorous-ab-readout/main.v` + `README.md`
- `examples/causal-did/main.v` + `README.md`
- `examples/churn-prediction/main.v` + `README.md`
- `examples/funnel-attribution/main.v` + `README.md`
- `examples/ratio-metric-inference/main.v` + `README.md`
- `examples/hypothesis-battery/main.v` + `README.md`

**Modify:**
- `Makefile` — add `docs` target
- `README.md` — add entry-path links
- `.gitignore` — add compiled binary patterns

**Delete:** `examples/*.v` (flat files), `examples/*.py`, compiled binaries

**Replaced by generated output:** all current `docs/*.html`, `docs/modules/*.html`

---

## Task 1: Build pipeline — `docs/build.py` + Makefile

**Files:**
- Create: `docs/build.py`
- Modify: `Makefile`

- [ ] **Step 1: Create `docs/build.py`**

```python
#!/usr/bin/env python3
"""
docs/build.py — Generate docs/**/*.html from docs/src/**/*.md.
Run from repo root: python docs/build.py
"""
import os, re, glob
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.toc import TocExtension

DOCS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DOCS_DIR)
SRC_DIR   = os.path.join(DOCS_DIR, 'src')
OUT_DIR   = DOCS_DIR

SIDEBAR_ROOT = """\
<ul class="nav-list">
  <li><a href="index.html">Home</a></li>
  <li><a href="getting-started.html">Getting Started</a></li>
  <li><a href="concepts.html">Concepts</a></li>
  <li><a href="examples.html">Examples</a></li>
</ul>
<span class="nav-section-label">Module Reference</span>
<ul class="nav-list">
  <li><a href="modules/stats.html">stats</a></li>
  <li><a href="modules/experiment.html">experiment</a></li>
  <li><a href="modules/ml.html">ml</a></li>
  <li><a href="modules/hypothesis.html">hypothesis</a></li>
  <li><a href="modules/growth.html">growth</a></li>
  <li><a href="modules/nn.html">nn</a></li>
  <li><a href="modules/prob.html">prob</a></li>
  <li><a href="modules/optim.html">optim</a></li>
  <li><a href="modules/linalg.html">linalg</a></li>
  <li><a href="modules/utils.html">utils</a></li>
</ul>
<span class="nav-section-label">Resources</span>
<ul class="nav-list">
  <li><a href="companion/index.html">Companion Docs</a></li>
</ul>"""

SIDEBAR_SUB = SIDEBAR_ROOT \
    .replace('href="index.html"', 'href="../index.html"') \
    .replace('href="getting-started.html"', 'href="../getting-started.html"') \
    .replace('href="concepts.html"', 'href="../concepts.html"') \
    .replace('href="examples.html"', 'href="../examples.html"') \
    .replace('href="modules/', 'href="../modules/') \
    .replace('href="companion/', 'href="../companion/')


def make_page(title, body, depth=0):
    prefix  = '../' * depth
    nav     = SIDEBAR_ROOT if depth == 0 else SIDEBAR_SUB
    logo_href = f'{prefix}index.html'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — VStats</title>
<link rel="stylesheet" href="{prefix}css/style.css">
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-header">
      <a href="{logo_href}" class="logo"><span>V</span>Stats</a>
    </div>
    <nav class="sidebar-nav">
      {nav}
    </nav>
  </aside>
  <main class="content">
    {body}
  </main>
</div>
<script src="{prefix}js/script.js"></script>
</body>
</html>"""


def expand_includes(text):
    def repl(m):
        rel  = m.group(1).strip()
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            return f'<!-- include not found: {rel} -->'
        content = open(path).read().rstrip()
        return f'```v\n{content}\n```'
    return re.sub(r'<!--\s*include:\s*([^\s>]+)\s*-->', repl, text)


def to_html(md_text):
    md = markdown.Markdown(extensions=[
        FencedCodeExtension(), TableExtension(), TocExtension(permalink=False),
    ])
    return md.convert(md_text)


def src_to_out(src_path):
    rel  = os.path.relpath(src_path, SRC_DIR)
    base = os.path.splitext(rel)[0] + '.html'
    return os.path.join(OUT_DIR, base)


def page_depth(out_path):
    rel = os.path.relpath(out_path, OUT_DIR)
    return len(rel.split(os.sep)) - 1


def build_all():
    srcs = sorted(glob.glob(os.path.join(SRC_DIR, '**', '*.md'), recursive=True))
    print(f'Building {len(srcs)} pages from docs/src/ ...')
    for src in srcs:
        text  = open(src).read()
        text  = expand_includes(text)
        body  = to_html(text)
        m     = re.search(r'<h1[^>]*>(.*?)</h1>', body)
        title = re.sub(r'<[^>]+>', '', m.group(1)) if m else 'VStats'
        out   = src_to_out(src)
        depth = page_depth(out)
        html  = make_page(title, body, depth=depth)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, 'w').write(html)
        print(f'  ✓ {os.path.relpath(src, REPO_ROOT)} → {os.path.relpath(out, REPO_ROOT)}')
    print(f'Done. {len(srcs)} pages built.')


if __name__ == '__main__':
    build_all()
```

- [ ] **Step 2: Add `docs` target to Makefile**

```makefile
.PHONY: test fulltest docs

test:
	v test tests/

fulltest:
	v -stats test tests/

docs:
	python docs/build.py
```

- [ ] **Step 3: Commit**

```bash
git add docs/build.py Makefile
git commit -m "feat(docs): add markdown build pipeline"
```

---

## Task 2: Docs source skeleton — stubs + smoke test

**Files:**
- Create: all `docs/src/*.md` and `docs/src/modules/*.md` as stubs

- [ ] **Step 1: Create stub files**

Create each file below with just its H1 title. They will be filled in later tasks.

`docs/src/index.md`:
```markdown
# VStats
```

`docs/src/getting-started.md`:
```markdown
# Getting Started
```

`docs/src/concepts.md`:
```markdown
# Concepts
```

`docs/src/examples.md`:
```markdown
# Examples
```

For each module, create `docs/src/modules/<name>.md` with content `# <name>`:
- `stats.md`, `experiment.md`, `ml.md`, `hypothesis.md`, `growth.md`
- `nn.md`, `prob.md`, `optim.md`, `linalg.md`, `utils.md`

- [ ] **Step 2: Run build and verify**

```bash
python docs/build.py
```

Expected output:
```
Building 14 pages from docs/src/ ...
  ✓ docs/src/index.md → docs/index.html
  ✓ docs/src/getting-started.md → docs/getting-started.html
  ...
Done. 14 pages built.
```

Then verify one generated file has the correct structure:
```bash
grep -c "sidebar" docs/index.html
```
Expected: `2` (sidebar div + nav class)

- [ ] **Step 3: Commit**

```bash
git add docs/src/
git commit -m "feat(docs): add markdown source skeleton"
```

---

## Task 3: `getting-started.md` — Python/R translation guide

**Files:**
- Modify: `docs/src/getting-started.md`

- [ ] **Step 1: Write full content**

Replace the stub with:

```markdown
# Getting Started (Python / R Users)

If you know what a t-test is, you're ready. No concept introductions here.

## Installation

```bash
v install https://github.com/rodabt/vstats
```

## Translation Table

| Python / scipy / R | vstats |
|--------------------|--------|
| `scipy.stats.ttest_ind(a, b)` | `experiment.abtest(ctrl, trt)` |
| `scipy.stats.mannwhitneyu(a, b)` | `hypothesis.mann_whitney_u_test(a, b)` |
| `scipy.stats.shapiro(x)` | `hypothesis.shapiro_wilk_test(x)` |
| `scipy.stats.chi2_contingency(table)` | `hypothesis.chi_squared_test(table)` |
| `statsmodels OLS + covariate` | `experiment.ancova(ctrl, trt, x_ctrl, x_trt)` |
| `statsmodels DiD interaction` | `experiment.did_regression(y, x, group, time)` |
| `statsmodels.stats.multitest.multipletests` | `stats.bh_correction(p_values, alpha)` |
| `scipy.stats.bootstrap` | `stats.bootstrap_test(ctrl, trt, n_resamples)` |
| `sklearn.linear_model.LogisticRegression` | `ml.logistic_regression(x, y, iters, lr)` |
| `sklearn.ensemble.RandomForestClassifier` | `ml.random_forest_classifier(x, y, n_trees, depth)` |
| `sklearn.metrics.roc_auc_score` | `utils.roc_curve(y_true, y_proba).auc` |
| `pandas` funnel groupby | `growth.create_funnel(stages, counts)` |
| Custom attribution logic | `growth.linear_attributes(touchpoints, conv, rev)` |

## Quick Examples

### Statistics

```v
import vstats.stats

x := [2.1, 2.4, 1.9, 2.3, 2.0, 2.2, 2.5, 1.8]
println(stats.mean(x))                           // 2.15
println(stats.standard_deviation(x))             // 0.23
println(stats.quantile(x, 0.75))                 // 2.4
w := stats.winsorize(x, 0.1, 0.9)               // cap extreme values
```

### Experimentation

```v
import vstats.experiment
import vstats.stats

ctrl := [10.1, 9.8, 10.2, 10.0, 10.3, 9.9, 10.1, 10.2, 9.7, 10.4]
trt  := [12.0, 11.8, 12.3, 12.1, 11.9, 12.2, 12.0, 11.7, 12.4, 12.1]

result := experiment.abtest(ctrl, trt)
println(experiment.null_verdict(result, 0.05))

// Multiple testing correction
p_vals := [result.p_value, 0.031, 0.21]
bh := stats.bh_correction(p_vals, 0.05)
println('Rejected: ${bh.n_rejected}/3')
```

### Machine Learning

```v
import vstats.utils
import vstats.ml

dataset := utils.load_breast_cancer()!
train, test := dataset.train_test_split(0.2)
x_tr, y_tr := train.xy()
x_te, y_te := test.xy()

x_tr_norm, mu, sigma := utils.normalize_features(x_tr)
x_te_norm := utils.apply_normalization(x_te, mu, sigma)

model := ml.random_forest_classifier(x_tr_norm, y_tr, 20, 5)
preds := ml.random_forest_predict(model, x_te_norm)
m := utils.binary_classification_metrics(y_te, preds)
println('F1: ${m["f1_score"]:.4f}')
```

### Growth Analytics

```v
import vstats.growth

funnel := growth.create_funnel(
    ['Landing', 'Signup', 'Purchase'],
    [10000, 3500, 420],
)
println('Conversion: ${funnel.conversion_rate * 100:.1f}%')
drop := funnel.highest_drop_off()
println('Worst step: ${drop.from_stage} → ${drop.to_stage}')
```

## Next Steps

- [Examples](examples.html) — six end-to-end scenarios
- [Module Reference](modules/stats.html) — full API
- [Companion Docs](companion/index.html) — statistical concepts
```

- [ ] **Step 2: Rebuild and spot-check**

```bash
python docs/build.py
grep -c "Translation Table" docs/getting-started.html
```
Expected: `1`

- [ ] **Step 3: Commit**

```bash
git add docs/src/getting-started.md docs/getting-started.html
git commit -m "docs: add getting-started guide for Python/R users"
```

---

## Task 4: `index.md` — two-path entry + navigation

**Files:**
- Modify: `docs/src/index.md`

- [ ] **Step 1: Write full content**

```markdown
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
```

- [ ] **Step 2: Rebuild**

```bash
python docs/build.py
grep "Getting Started" docs/index.html | head -2
```
Expected: two matches (sidebar link + hero link)

- [ ] **Step 3: Commit**

```bash
git add docs/src/index.md docs/index.html
git commit -m "docs: rewrite index with two-path entry"
```

---

## Task 5: `concepts.md` — migrate from existing HTML

**Files:**
- Modify: `docs/src/concepts.md`

- [ ] **Step 1: Write `docs/src/concepts.md`**

```markdown
# Core Concepts

## Linear Algebra Basics

Linear algebra provides the mathematical foundation for machine learning. Vectors represent
points in n-dimensional space, while matrices represent linear transformations.

### Vectors

A vector is an ordered collection of numbers. Key operations include:

- **Addition:** Element-wise sum of two vectors
- **Dot Product:** a · b = sum of element-wise products
- **Magnitude:** Length of vector (L2 norm)

### Matrices

A matrix is a 2D array with rows and columns. Key operations:

- **Multiplication:** Row-column dot products
- **Transpose:** Swap rows and columns
- **Identity:** Matrix that does nothing (1s on diagonal)

## Statistics Fundamentals

### Descriptive Statistics

- **Mean:** Arithmetic average of all values
- **Median:** Middle value (50th percentile)
- **Variance:** Average squared deviation from mean (vstats uses *sample* variance, ÷n-1)
- **Standard Deviation:** Square root of variance

### Hypothesis Testing

Statistical tests determine if observed differences are significant:

- **t-test:** Compare means of two groups
- **ANOVA:** Compare means of 3+ groups
- **p-value:** Probability of observing results by chance under the null hypothesis

## Machine Learning Essentials

### Supervised vs Unsupervised

- **Supervised:** Learn from labeled data (regression, classification)
- **Unsupervised:** Find patterns in unlabeled data (clustering)

### Model Evaluation

- **MSE:** Mean squared error (regression)
- **Accuracy:** Correct predictions / Total
- **R²:** Proportion of variance explained

## Growth Metrics

| Metric | Formula |
|--------|---------|
| ARPA | Revenue / Accounts |
| CAC | Acquisition Spend / New Customers |
| LTV | ARPU × Customer Lifespan |
| Churn Rate | Customers Lost / Total Customers |
| NRR | (MRR_end − Churn_MRR) / MRR_start |

## Experimentation

### Statistical Power and Sample Size

Before running an experiment, you must decide how many observations you need. Four numbers
determine this:

- **Alpha (α):** The false positive rate — how often you'll declare a winner when there's no
  real effect. Typically 0.05 (5%).
- **Power (1 − β):** The probability of detecting a real effect when one exists. Typically
  0.80 (80%). Setting power too low means your experiment may end with no result even when
  the treatment works.
- **Minimum Detectable Effect (MDE):** The smallest improvement worth detecting. A tighter
  MDE requires more data. Be honest about what effect size would change a business decision
  — don't chase tiny effects that aren't actionable.
- **Baseline variance:** Higher variance in your metric means more noise, requiring more
  data to see signal. Use historical data to estimate this.

The formula for continuous metrics: `n = 2 × ((z_α/2 + z_β) × σ / Δ)²` per group, where
σ is the standard deviation and Δ is the MDE.

### Frequentist vs. Bayesian

Both are valid frameworks with different outputs:

- **Frequentist (p-values):** Answers "if there were no effect, how unlikely is this data?"
  You reject the null hypothesis when p < α. The result is binary: significant or not. Does
  not give the probability that B is better.
- **Bayesian (posteriors):** Answers "given this data, what is the probability that B beats
  A?" More intuitive for business decisions. Requires specifying a prior — use Beta(1,1)
  (uniform) when you have no prior knowledge.

Bayesian is preferable when you need to communicate results to non-statisticians ("94%
chance B is better"), when you want to incorporate prior knowledge, or when you need to
make a decision before collecting enough data for a frequentist test.

### The Peeking Problem

A common mistake: checking an experiment's p-value every day and stopping when it first
crosses 0.05. This *inflates the false positive rate dramatically* — you may declare
significance by chance, especially early in an experiment.

Two solutions:

- **Sequential testing (SPRT):** Uses Wald's likelihood ratio test with boundaries that
  account for repeated looks. You can check at any time; the false positive rate stays at α.
  Use `experiment.sprt_test`.
- **Pre-commit to a fixed end date:** Calculate your sample size, run the experiment until
  you have it, then do one final test. Never look at results before the end.

### Effect Sizes

A p-value tells you whether an effect exists; effect size tells you how big it is. Always
report both.

- **Cohen's d:** For continuous metrics. d = (mean_B − mean_A) / pooled_std. Small: 0.2,
  Medium: 0.5, Large: 0.8.
- **Absolute lift:** For proportions. "The signup rate increased by 1.2 percentage points
  (from 5.0% to 6.2%)." Always prefer absolute over relative for communication.
- **Relative lift:** Percentage change relative to control. Useful for comparing across
  metrics with different baselines, but can be misleading for small baselines.

### Variance Reduction with CUPED

CUPED (Controlled-experiment Using Pre-Experiment Data) exploits the correlation between a
user's pre-experiment behavior and their in-experiment behavior. If last week's revenue
predicts this week's revenue, you can "remove" that predictable component from both groups,
reducing noise without introducing bias. A correlation of ρ = 0.7 reduces required sample
size by approximately 1 − ρ² = 51%.
```

- [ ] **Step 3: Rebuild and verify section count**

```bash
python docs/build.py
grep -c "<h2>" docs/concepts.html
```
Expected: `5` (five top-level sections)

- [ ] **Step 4: Commit**

```bash
git add docs/src/concepts.md docs/concepts.html
git commit -m "docs: migrate concepts to markdown"
```

---

## Task 6: Module pages — migrate all 10 modules

**Files:**
- Modify: all `docs/src/modules/*.md`

Each module page follows this pattern — shown in full for `stats.md`, then abbreviated for the rest.

- [ ] **Step 1: Write `docs/src/modules/stats.md`**

```markdown
# stats

`import vstats.stats`

Descriptive statistics, outlier handling, multiple testing corrections, and
ratio-metric inference.

> **vs Python:** `stats.mean`, `stats.variance`, `stats.correlation` replace
> `numpy.mean`, `numpy.var`, `scipy.stats.pearsonr`. Functions are generic —
> they accept `[]int` or `[]f64` and return `f64`.

## Descriptive

```v
sum[T](x []T) T
mean[T](x []T) f64
median(x []f64) f64                        // requires []f64
quantile(x []f64, p f64) f64               // uses int(p*n) truncation, not rounding
mode(x []f64) []f64
range[T](x []T) T
variance[T](x []T) f64                     // sample variance (÷n-1)
standard_deviation[T](x []T) f64
interquartile_range(x []f64) f64
covariance[T](x []T, y []T) f64
correlation[T](x []T, y []T) f64
dev_mean[T](x []T) []f64
```

## Effect Sizes & Tests

```v
cohens_d[T](group1 []T, group2 []T) f64
cramers_v(contingency [][]int) f64
skewness[T](x []T) f64
kurtosis[T](x []T) f64
anova_one_way[T](groups [][]T) (f64, f64)               // (F-stat, p-value)
confidence_interval_mean[T](x []T, level f64) (f64, f64)
```

## Outlier Handling

> **vs Python:** `stats.winsorize` replaces `scipy.stats.mstats.winsorize`.
> `stats.rtm_correction` has no direct scipy equivalent.

```v
winsorize(x []f64, q_low f64, q_high f64) []f64
rtm_correction(baseline []f64, followup []f64, selection_threshold f64) f64
```

## Multiple Testing

> **vs Python:** replaces `statsmodels.stats.multitest.multipletests`.

```v
bh_correction(p_values []f64, alpha f64) BHResult
// BHResult{ adjusted []f64, reject []bool, n_rejected int }

bonferroni_correction(p_values []f64, alpha f64) BonferroniResult
```

## Ratio Metrics & Bootstrap

> **vs Python:** `delta_method_ratio` replaces manual linearization + `scipy.stats.ttest_ind`.
> `bootstrap_test` replaces `scipy.stats.bootstrap`.

```v
delta_method_ratio(a []f64, b []f64, treatment []int, cfg DeltaMethodConfig) DeltaMethodResult
// DeltaMethodConfig{ alpha f64 = 0.05 }
// DeltaMethodResult{ ratio_ctrl, ratio_trt, effect, se, t_statistic, p_value, ci_lower, ci_upper f64 }

bootstrap_test(ctrl []f64, trt []f64, n_resamples int) BootstrapResult
// BootstrapResult{ p_value, observed_diff, ci_lower, ci_upper f64; n_resamples int }
```

## See Also

- [ratio-metric-inference example](../examples.html#ratio-metric-inference)
- [rigorous-ab-readout example](../examples.html#rigorous-ab-readout)
```

- [ ] **Step 2: Write the remaining 9 module pages**

For each module below, create `docs/src/modules/<name>.md` following the same pattern:
`# <Module>`, import line, one-sentence description, `> **vs Python:**` callout per section,
then the function signatures (copy from `~/.claude/skills/vstats/references/modules.md`).

Modules: `experiment`, `ml`, `hypothesis`, `growth`, `nn`, `prob`, `optim`, `linalg`, `utils`

For `experiment.md`, include sections for: A/B Testing, Sample Size, Readout Checks, DiD, PSM.
For `ml.md`, group by: Regression, Classification, Clustering.
For `hypothesis.md`, note that all tests return `(statistic f64, p_value f64)`.

- [ ] **Step 3: Rebuild and verify**

```bash
python docs/build.py
ls docs/modules/
```
Expected: 10 `.html` files

- [ ] **Step 4: Commit**

```bash
git add docs/src/modules/ docs/modules/
git commit -m "docs: migrate all module pages to markdown with vs-Python callouts"
```

---

## Task 7: `examples/rigorous-ab-readout/`

**Files:**
- Create: `examples/rigorous-ab-readout/main.v`
- Create: `examples/rigorous-ab-readout/README.md`

- [ ] **Step 1: Create `main.v`**

```v
// Scenario: Rigorous A/B Test Readout
// Demonstrates: vstats.experiment + vstats.stats
// Python equivalent: scipy.stats + statsmodels + custom SRM + multipletests = 5 imports, ~80 lines
module main

import vstats.experiment
import vstats.stats

fn main() {
	println('=== Rigorous A/B Test Readout ===\n')

	// --- Setup: checkout-flow experiment, 20 users per arm ---
	// Revenue per user (0 = non-purchaser). Pre-period used for CUPED.
	pre_ctrl := [40.0, 0, 0, 50.0, 0, 45.0, 0, 0, 0, 48.0,
	             42.0, 0, 0, 0, 49.0, 0, 44.0, 0, 0, 47.0]
	pre_trt  := [43.0, 0, 53.0, 0, 48.0, 0, 0, 45.0, 0, 51.0,
	             44.0, 0, 0, 54.0, 0, 49.0, 0, 0, 0, 50.0]
	ctrl     := [42.0, 0, 0, 55.0, 0, 48.0, 0, 0, 0, 51.0,
	             44.0, 0, 0, 0, 52.0, 0, 47.0, 0, 0, 49.0]
	trt      := [45.0, 0, 56.0, 0, 50.0, 0, 0, 48.0, 0, 53.0,
	             46.0, 0, 0, 57.0, 0, 51.0, 0, 0, 0, 52.0]

	// --- Core analysis ---

	// 1. SRM: verify assignment matches expected 50/50 split
	srm := experiment.srm_test(ctrl.len, trt.len, 0.5, 0.05)
	println('1. SRM check')
	println('   chi2=${srm.chi2_statistic:.4f}  p=${srm.p_value:.4f}  detected=${srm.srm_detected}')

	// 2. Winsorize to cap heavy tails before analysis
	ctrl_w := stats.winsorize(ctrl, 0.05, 0.95)
	trt_w  := stats.winsorize(trt, 0.05, 0.95)

	// 3. CUPED: use pre-period data to reduce variance
	cuped := experiment.cuped_test(ctrl_w, trt_w, pre_ctrl, pre_trt)
	println('\n2. CUPED')
	println('   theta=${cuped.theta:.4f}  variance_reduction=${cuped.variance_reduction * 100:.1f}%')
	result := cuped.adjusted_result

	// 4. BH correction across three metrics tested simultaneously
	p_revenue  := result.p_value
	p_sessions := 0.031   // second metric tested in same experiment
	p_bounce   := 0.210   // third metric
	bh := stats.bh_correction([p_revenue, p_sessions, p_bounce], 0.05)
	println('\n3. Multiple testing (BH, 3 metrics)')
	println('   revenue  adj_p=${bh.adjusted[0]:.4f}  reject=${bh.reject[0]}')
	println('   sessions adj_p=${bh.adjusted[1]:.4f}  reject=${bh.reject[1]}')
	println('   bounce   adj_p=${bh.adjusted[2]:.4f}  reject=${bh.reject[2]}')
	println('   total rejected: ${bh.n_rejected}/3')

	// --- Interpret output ---
	println('\n4. Verdict')
	println('   ' + experiment.null_verdict(result, 0.05))
}
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Rigorous A/B Test Readout

End-to-end experiment analysis pipeline covering the four checks a rigorous
readout requires: SRM detection, outlier handling, variance reduction (CUPED),
and multiple testing correction (Benjamini-Hochberg).

**Run:** `v run examples/rigorous-ab-readout/main.v`

**Modules used:** `vstats.experiment`, `vstats.stats`

**Python equivalent:** `scipy.stats` + `statsmodels` + `statsmodels.stats.multitest`
+ custom SRM logic — four separate libraries, ~80 lines.
```

- [ ] **Step 3: Verify it runs**

```bash
v run examples/rigorous-ab-readout/main.v
```

Expected: output with SRM check, CUPED variance reduction %, BH rejection counts, verdict line. Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add examples/rigorous-ab-readout/
git commit -m "feat(examples): add rigorous-ab-readout scenario"
```

---

## Task 8: `examples/causal-did/`

**Files:**
- Create: `examples/causal-did/main.v`
- Create: `examples/causal-did/README.md`

- [ ] **Step 1: Create `main.v`**

```v
// Scenario: Causal Difference-in-Differences
// Demonstrates: vstats.experiment — DiD regression + parallel trends test
// Python equivalent: statsmodels OLS with interaction term + manual trend test
module main

import vstats.experiment

fn main() {
	println('=== Causal DiD: Policy Impact Analysis ===\n')
	println('Setting: two regions, 10 units each. Treatment region receives new')
	println('pricing in period 2. True DiD effect: +3.0 units.\n')

	// --- Setup ---
	// Pre-period data for parallel trends test
	y_treat_pre := [10.0, 9.9, 10.2, 10.1, 9.8, 10.3, 9.7, 10.2, 10.0, 9.9]
	y_ctrl_pre  := [9.8,  10.1, 10.3, 9.9, 10.2, 10.0, 9.7, 10.4, 10.1, 9.9]
	time_pre    := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	// Full panel: pre + post for both groups
	mut y     := []f64{}
	mut group := []int{}  // 0=control, 1=treated
	mut time  := []int{}  // 0=pre, 1=post

	for v in y_ctrl_pre  { y << v; group << 0; time << 0 }
	// Control post: common trend +2
	for v in [11.9, 12.1, 12.2, 11.8, 12.3, 12.0, 11.7, 12.4, 12.1, 11.9] {
		y << v; group << 0; time << 1
	}
	for v in y_treat_pre { y << v; group << 1; time << 0 }
	// Treated post: common trend +2 plus treatment effect +3 = +5 total
	for v in [14.8, 15.1, 15.3, 14.9, 15.2, 15.0, 14.7, 15.4, 15.1, 14.9] {
		y << v; group << 1; time << 1
	}

	// --- Core analysis ---
	cfg := experiment.DiDConfig{}

	// 1. Parallel trends test (pre-period only — assumption check)
	trends := experiment.test_parallel_trends(y_treat_pre, y_ctrl_pre, time_pre, cfg)
	println('1. Parallel trends test (pre-period assumption check)')
	println('   slope_treated=${trends.slope_treated:.4f}  slope_control=${trends.slope_control:.4f}')
	println('   difference p=${trends.p_value:.4f}  holds=${trends.parallel_trends_hold}')

	// 2. DiD regression (OLS with treatment × post interaction)
	did := experiment.did_regression(y, [][]f64{}, group, time, cfg)
	println('\n2. DiD regression')
	println('   effect=${did.did_coefficient:.4f}  (true: 3.0)')
	println('   SE=${did.did_se:.4f}  t=${did.did_t_stat:.4f}  p=${did.did_p_value:.4f}')
	println('   95% CI: [${did.did_ci_lower:.4f}, ${did.did_ci_upper:.4f}]')
	println('   R²=${did.r_squared:.4f}')

	// --- Interpret output ---
	println('\n3. Verdict')
	if did.did_p_value < 0.05 {
		println('   Significant causal effect: DiD = ${did.did_coefficient:.3f}')
	} else {
		println('   No significant causal effect detected.')
	}
}
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Causal Difference-in-Differences

Estimates a causal treatment effect from observational panel data using DiD
regression. Includes the parallel trends assumption check as a pre-analysis step.

**Run:** `v run examples/causal-did/main.v`

**Modules used:** `vstats.experiment`

**Python equivalent:** `statsmodels.formula.api.ols('y ~ group * time', data).fit()`
plus manual parallel trends slope comparison.
```

- [ ] **Step 3: Verify it runs**

```bash
v run examples/causal-did/main.v
```

Expected: parallel trends p > 0.05 (trends hold), DiD ≈ 3.0, significant. Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add examples/causal-did/
git commit -m "feat(examples): add causal-did scenario"
```

---

## Task 9: `examples/churn-prediction/`

This scenario consolidates the old `iris_logistic_classification.v`, `breast_cancer_classification.v`, and `titanic_*.v` files.

**Files:**
- Create: `examples/churn-prediction/main.v`
- Create: `examples/churn-prediction/README.md`

- [ ] **Step 1: Create `main.v`**

```v
// Scenario: Customer Churn Prediction
// Demonstrates: vstats.ml + vstats.utils — full binary classification pipeline
// Python equivalent: sklearn.ensemble.RandomForestClassifier + classification_report
module main

import vstats.utils
import vstats.ml

fn main() {
	println('=== Customer Churn Prediction ===\n')
	println('Using Breast Cancer dataset (malignant=churned, benign=retained).\n')

	// --- Setup ---
	dataset := utils.load_breast_cancer()!
	train, test := dataset.train_test_split(0.2)
	x_train, y_train := train.xy()
	x_test, y_test   := test.xy()

	// Normalize on train only — no leakage into test
	x_train_norm, feat_mean, feat_std := utils.normalize_features(x_train)
	x_test_norm := utils.apply_normalization(x_test, feat_mean, feat_std)

	println('Train: ${x_train.len} samples  Test: ${x_test.len} samples  Features: ${x_train[0].len}\n')

	// --- Core analysis ---

	// Baseline: logistic regression
	lr := ml.logistic_regression(x_train_norm, y_train.map(f64(it)), 200, 0.1)
	lr_pred := ml.logistic_predict(lr, x_test_norm, 0.5).map(int(it))
	lr_m := utils.binary_classification_metrics(y_test, lr_pred)
	println('Logistic Regression')
	println('  accuracy=${lr_m["accuracy"]:.4f}  precision=${lr_m["precision"]:.4f}  recall=${lr_m["recall"]:.4f}  f1=${lr_m["f1_score"]:.4f}')

	// Random Forest
	rf := ml.random_forest_classifier(x_train_norm, y_train, 20, 5)
	rf_pred := ml.random_forest_predict(rf, x_test_norm)
	rf_m := utils.binary_classification_metrics(y_test, rf_pred)
	println('\nRandom Forest (20 trees, max_depth=5)')
	println('  accuracy=${rf_m["accuracy"]:.4f}  precision=${rf_m["precision"]:.4f}  recall=${rf_m["recall"]:.4f}  f1=${rf_m["f1_score"]:.4f}')

	// ROC / AUC
	rf_proba := ml.random_forest_classifier_predict_proba(rf, x_test_norm)
	roc := utils.roc_curve(y_test, rf_proba)
	println('\nRandom Forest AUC: ${roc.auc:.4f}')

	// --- Interpret output ---
	println('\nConfusion matrix (Random Forest):')
	cm := utils.build_confusion_matrix(y_test, rf_pred)
	println(cm.summary())
}
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Customer Churn Prediction

Full binary classification pipeline: load data, normalize, train logistic regression
and random forest, evaluate with precision/recall/F1 and AUC.

**Verify (do not run — slow):** `v build examples/churn-prediction/main.v`

**Modules used:** `vstats.ml`, `vstats.utils`

**Python equivalent:** `sklearn` pipeline with `LogisticRegression`,
`RandomForestClassifier`, `classification_report`, `roc_auc_score`.
```

- [ ] **Step 3: Verify it compiles (do not run)**

```bash
v build examples/churn-prediction/main.v
```

Expected: exit code 0, no errors. Do not run — this takes 30+ seconds.

- [ ] **Step 4: Commit**

```bash
git add examples/churn-prediction/
git commit -m "feat(examples): add churn-prediction scenario"
```

---

## Task 10: `examples/funnel-attribution/`

**Files:**
- Create: `examples/funnel-attribution/main.v`
- Create: `examples/funnel-attribution/README.md`

- [ ] **Step 1: Create `main.v`**

```v
// Scenario: Funnel Drop-off + Multi-Touch Attribution
// Demonstrates: vstats.growth — funnel analysis and marketing attribution
// Python equivalent: custom pandas aggregation; no standard library covers attribution
module main

import vstats.growth

fn main() {
	println('=== E-Commerce Funnel + Marketing Attribution ===\n')

	// --- Setup: 4-stage checkout funnel ---
	stages := ['Landing Page', 'Product View', 'Add to Cart', 'Purchase']
	users  := [10000, 4200, 1800, 540]

	// --- Core analysis ---

	// 1. Funnel conversion and drop-off
	funnel := growth.create_funnel(stages, users)
	println('1. Funnel Analysis')
	println('   Overall conversion: ${funnel.conversion_rate * 100:.2f}%')
	drop := funnel.highest_drop_off()
	println('   Worst drop-off: ${drop.from_stage} → ${drop.to_stage}')

	// 2. A/B test two funnel variants
	funnel_b := growth.create_funnel(stages, [10000, 4800, 2200, 680])
	winner := growth.ab_test_funnel(funnel, funnel_b)
	result_str := if winner == 1 { 'B wins' } else if winner == -1 { 'A wins' } else { 'no significant difference' }
	println('   A vs B: ${result_str}')

	// 3. Last-touch attribution
	channels    := ['paid_search', 'email', 'organic', 'social', 'paid_search',
	                'email', 'organic', 'social', 'direct', 'direct']
	conversions := [true, false, true, false, true, true, false, false, true, true]
	revenue     := [120.0, 0, 85.0, 0, 200.0, 95.0, 0, 0, 150.0, 75.0]

	println('\n2. Last-touch Attribution')
	last := growth.last_touch_attributes(channels, conversions, revenue)
	for r in last {
		println('   ${r.channel:-15s}  conversions=${r.conversions}  revenue=\$${r.revenue:.0f}')
	}

	// 4. Multi-touch linear attribution
	touchpoints := [
		['paid_search', 'email'],
		['email'],
		['organic', 'direct'],
		['social'],
		['paid_search', 'social', 'email'],
		['email', 'direct'],
		['organic'],
		['social', 'email'],
		['direct'],
		['direct'],
	]
	println('\n3. Linear (multi-touch) Attribution')
	linear := growth.linear_attributes(touchpoints, conversions, revenue)
	for r in linear {
		println('   ${r.channel:-15s}  conversions=${r.conversions:.2f}  revenue=\$${r.revenue:.0f}')
	}

	// 5. Channel ROI
	costs := {
		'paid_search': 500.0
		'email':        50.0
		'organic':       0.0
		'social':       200.0
		'direct':        0.0
	}
	println('\n4. Channel ROI (last-touch)')
	roi := growth.channel_roi(last, costs)
	for ch, r in roi {
		println('   ${ch:-15s}  ROI=${r:.2f}x')
	}
}
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Funnel Drop-off + Multi-Touch Attribution

Funnel conversion analysis, A/B testing of funnel variants, and marketing
attribution (last-touch and linear multi-touch) with channel ROI calculation.

**Run:** `v run examples/funnel-attribution/main.v`

**Modules used:** `vstats.growth`

**Python equivalent:** custom pandas groupby for funnels; no standard library
covers multi-touch attribution natively.
```

- [ ] **Step 3: Verify it runs**

```bash
v run examples/funnel-attribution/main.v
```

Expected: funnel conversion %, attribution table, ROI per channel. Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add examples/funnel-attribution/
git commit -m "feat(examples): add funnel-attribution scenario"
```

---

## Task 11: `examples/ratio-metric-inference/`

**Files:**
- Create: `examples/ratio-metric-inference/main.v`
- Create: `examples/ratio-metric-inference/README.md`

- [ ] **Step 1: Create `main.v`**

```v
// Scenario: Ratio Metric Inference (Revenue per Session)
// Demonstrates: vstats.stats — delta method + permutation bootstrap
// Python equivalent: manual linearization + scipy.stats.ttest_ind + scipy.stats.bootstrap
module main

import vstats.experiment
import vstats.stats
import rand

fn main() {
	println('=== Ratio Metric Inference: Revenue per Session ===\n')
	println('Problem: a naive t-test on revenue/session is biased because the')
	println('numerator (revenue) and denominator (sessions) are correlated per user.')
	println('The delta method linearizes the ratio before testing.\n')

	rand.seed([u32(42), u32(0)])

	// --- Setup: 10 users per arm ---
	// Control: ~$5/session. Treatment: ~$7/session (+$2 lift).
	revenue_ctrl  := [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 7.0, 13.0, 10.0, 14.0]
	sessions_ctrl := [2.0,  2.0,  2.0,  3.0,  2.0,  2.0,  1.0,  3.0,  2.0,  2.0]
	revenue_trt   := [14.0, 16.0, 12.0, 21.0, 13.0, 15.0, 7.0, 19.0, 14.0, 18.0]
	sessions_trt  := [2.0,  2.0,  2.0,  3.0,  2.0,  2.0,  1.0,  3.0,  2.0,  2.0]

	mut revenue   := []f64{}
	mut sessions  := []f64{}
	mut treatment := []int{}
	for v in revenue_ctrl  { revenue << v }
	for v in revenue_trt   { revenue << v }
	for v in sessions_ctrl { sessions << v }
	for v in sessions_trt  { sessions << v }
	for _ in revenue_ctrl  { treatment << 0 }
	for _ in revenue_trt   { treatment << 1 }

	// --- Core analysis ---

	// 1. Naive t-test on raw revenue (biased for ratio metrics)
	naive := experiment.abtest(revenue_ctrl, revenue_trt)
	println('1. Naive t-test on raw revenue (ignores session denominator)')
	println('   ctrl_mean=${naive.control_mean:.2f}  trt_mean=${naive.treatment_mean:.2f}  p=${naive.p_value:.4f}')

	// 2. Delta method (correct approach for ratio metrics)
	dm := stats.delta_method_ratio(revenue, sessions, treatment)
	println('\n2. Delta method (correct for revenue/session)')
	println('   ctrl_ratio=${dm.ratio_ctrl:.4f}  trt_ratio=${dm.ratio_trt:.4f}')
	println('   effect=${dm.effect:.4f}  SE=${dm.se:.4f}')
	println('   t=${dm.t_statistic:.4f}  p=${dm.p_value:.4f}')
	println('   95% CI: [${dm.ci_lower:.4f}, ${dm.ci_upper:.4f}]')

	// 3. Permutation bootstrap on linearized residuals — non-parametric check
	r_global := stats.mean(revenue) / stats.mean(sessions)
	mut z_ctrl := []f64{}
	mut z_trt  := []f64{}
	for i in 0 .. revenue_ctrl.len {
		z_ctrl << revenue_ctrl[i] - r_global * sessions_ctrl[i]
		z_trt  << revenue_trt[i]  - r_global * sessions_trt[i]
	}
	boot := stats.bootstrap_test(z_ctrl, z_trt, 2000)
	println('\n3. Permutation bootstrap (non-parametric robustness check)')
	println('   observed_diff=${boot.observed_diff:.4f}  p=${boot.p_value:.4f}')
	println('   95% CI: [${boot.ci_lower:.4f}, ${boot.ci_upper:.4f}]')

	// --- Interpret output ---
	println('\n4. Conclusion')
	if dm.p_value < 0.05 {
		println('   Delta method detects significant lift: +${dm.effect:.2f} revenue/session.')
	} else {
		println('   No significant lift in revenue/session.')
	}
}
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Ratio Metric Inference

Demonstrates why a naive t-test on ratio metrics (revenue/session) is biased,
and how the delta method correctly linearizes the ratio before testing.
Includes a permutation bootstrap as a non-parametric robustness check.

**Run:** `v run examples/ratio-metric-inference/main.v`

**Modules used:** `vstats.stats`, `vstats.experiment`

**Python equivalent:** manual linearization + `scipy.stats.ttest_ind` +
`scipy.stats.bootstrap` — three separate steps with no built-in ratio support.
```

- [ ] **Step 3: Verify it runs**

```bash
v run examples/ratio-metric-inference/main.v
```

Expected: naive p-value, delta method ratio estimates, bootstrap CI. Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add examples/ratio-metric-inference/
git commit -m "feat(examples): add ratio-metric-inference scenario"
```

---

## Task 12: `examples/hypothesis-battery/`

**Files:**
- Create: `examples/hypothesis-battery/main.v`
- Create: `examples/hypothesis-battery/README.md`

- [ ] **Step 1: Create `main.v`**

```v
// Scenario: Hypothesis Testing Battery
// Demonstrates: vstats.hypothesis — normality check → parametric/non-parametric decision
// Python equivalent: scipy.stats.shapiro + ttest_ind + mannwhitneyu (same functions, same pattern)
module main

import vstats.hypothesis
import vstats.stats

fn main() {
	println('=== Hypothesis Testing Battery ===\n')
	println('Pattern: check normality first, then choose parametric vs non-parametric.\n')

	tp := hypothesis.TestParams{ alpha: 0.05 }

	// --- Scenario A: near-normal groups, clear effect ---
	a1 := [10.1, 9.8, 10.2, 10.0, 10.3, 9.9, 10.1, 10.2, 9.7, 10.4]
	a2 := [12.0, 11.8, 12.3, 12.1, 11.9, 12.2, 12.0, 11.7, 12.4, 12.1]

	println('--- Scenario A: near-normal groups ---')
	_, p_sw_a1 := hypothesis.shapiro_wilk_test(a1)
	_, p_sw_a2 := hypothesis.shapiro_wilk_test(a2)
	println('Shapiro-Wilk: group1 p=${p_sw_a1:.4f}  group2 p=${p_sw_a2:.4f}')

	if p_sw_a1 > 0.05 && p_sw_a2 > 0.05 {
		println('Both normal → Welch t-test')
		t, p_t := hypothesis.t_test_two_sample(a1, a2, tp)
		d := stats.cohens_d(a2, a1)
		println('t=${t:.4f}  p=${p_t:.4f}  Cohen\'s d=${d:.4f}')
	} else {
		println('Non-normal → Mann-Whitney U')
		u, p_mw := hypothesis.mann_whitney_u_test(a1, a2)
		println('U=${u:.4f}  p=${p_mw:.4f}')
	}

	// --- Scenario B: skewed groups with outlier ---
	b1 := [1.0, 1.2, 1.1, 8.0, 1.3, 1.0, 1.2, 1.1, 1.0, 1.3]
	b2 := [1.5, 1.6, 1.4, 9.0, 1.7, 1.5, 1.6, 1.4, 1.5, 1.7]

	println('\n--- Scenario B: skewed groups (outlier present) ---')
	_, p_sw_b1 := hypothesis.shapiro_wilk_test(b1)
	_, p_sw_b2 := hypothesis.shapiro_wilk_test(b2)
	println('Shapiro-Wilk: group1 p=${p_sw_b1:.4f}  group2 p=${p_sw_b2:.4f}')

	if p_sw_b1 > 0.05 && p_sw_b2 > 0.05 {
		println('Both normal → Welch t-test')
		t, p_t := hypothesis.t_test_two_sample(b1, b2, tp)
		println('t=${t:.4f}  p=${p_t:.4f}')
	} else {
		println('Non-normal → Mann-Whitney U (robust to outliers)')
		u, p_mw := hypothesis.mann_whitney_u_test(b1, b2)
		println('U=${u:.4f}  p=${p_mw:.4f}')
	}

	// --- Summary ---
	println('\n--- Decision rule ---')
	println('Shapiro-Wilk p > 0.05 → data is normal → use t-test')
	println('Shapiro-Wilk p ≤ 0.05 → non-normal → use Mann-Whitney U')
	println('With n > 100, CLT makes t-test robust regardless of normality.')
}
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Hypothesis Testing Battery

Demonstrates the normality-first decision pattern: run Shapiro-Wilk, then choose
Welch t-test (normal) or Mann-Whitney U (non-normal). Applied to two scenarios —
clean data and outlier-heavy data — to show when the choice matters.

**Run:** `v run examples/hypothesis-battery/main.v`

**Modules used:** `vstats.hypothesis`, `vstats.stats`

**Python equivalent:** `scipy.stats.shapiro` + `scipy.stats.ttest_ind` +
`scipy.stats.mannwhitneyu` — same pattern, same three functions.
```

- [ ] **Step 3: Verify it runs**

```bash
v run examples/hypothesis-battery/main.v
```

Expected: Scenario A uses t-test, Scenario B uses Mann-Whitney. Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add examples/hypothesis-battery/
git commit -m "feat(examples): add hypothesis-battery scenario"
```

---

## Task 13: Wire `examples.md` with include tags

**Files:**
- Modify: `docs/src/examples.md`

- [ ] **Step 1: Write full content**

```markdown
# Examples

Six end-to-end scenarios, each targeting a different module and showcasing
something you can't do in one call with scipy or sklearn.

All examples are runnable: `v run examples/<scenario>/main.v`
(exception: `churn-prediction` — verify with `v build` only, runtime is slow)

---

## rigorous-ab-readout

SRM check + winsorization + CUPED variance reduction + Benjamini-Hochberg
correction + plain-English verdict — the full rigor checklist in one pipeline.

<!-- include: examples/rigorous-ab-readout/main.v -->

---

## causal-did

Parallel trends assumption check followed by DiD regression with OLS standard
errors — causal inference from panel data without statsmodels.

<!-- include: examples/causal-did/main.v -->

---

## churn-prediction

Full binary classification pipeline: normalize, train logistic regression and
random forest, evaluate with F1 and AUC. (Verify with `v build` — slow to run.)

<!-- include: examples/churn-prediction/main.v -->

---

## funnel-attribution

Funnel drop-off analysis, A/B test of funnel variants, last-touch and linear
multi-touch attribution, channel ROI — all from one `growth` import.

<!-- include: examples/funnel-attribution/main.v -->

---

## ratio-metric-inference

Why naive t-tests are wrong for revenue/session metrics, and how the delta method
and permutation bootstrap give correct inference on ratio metrics.

<!-- include: examples/ratio-metric-inference/main.v -->

---

## hypothesis-battery

Normality check (Shapiro-Wilk) → parametric (Welch t-test) or non-parametric
(Mann-Whitney U) decision, applied to clean and outlier-heavy scenarios.

<!-- include: examples/hypothesis-battery/main.v -->
```

- [ ] **Step 2: Rebuild and verify includes expanded**

```bash
python docs/build.py
grep -c "module main" docs/examples.html
```

Expected: `6` (one `module main` per embedded scenario)

- [ ] **Step 3: Commit**

```bash
git add docs/src/examples.md docs/examples.html
git commit -m "docs: wire examples.md with include tags for all 6 scenarios"
```

---

## Task 14: Cleanup — delete old files, update README + .gitignore

**Files:**
- Delete: `examples/*.v` (old flat files), `examples/*.py`, compiled binaries
- Modify: `README.md`, `.gitignore`

- [ ] **Step 1: Delete old example files**

```bash
git rm examples/breast_cancer_classification.v
git rm examples/breast_cancer_classification.py
git rm examples/iris_logistic_classification.v
git rm examples/iris_logistic_classification.py
git rm examples/titanic_logistic_test.v
git rm examples/titanic_naive_bayes_test.v
git rm examples/titanic_random_forest_test.v
git rm examples/generic_types_example.v
```

- [ ] **Step 2: Add compiled binaries to `.gitignore`**

Add these lines to `.gitignore` (the compiled binaries sitting in `examples/`):

```
examples/breast_cancer_classification
examples/iris_logistic_classification
examples/titanic_random_forest_test
examples/generic_types_example
examples/*/main
```

- [ ] **Step 3: Update `README.md`**

Replace the Documentation section with:

```markdown
## Documentation

- **[docs/index.html](docs/index.html)** — full API reference, concepts, examples
- **[docs/getting-started.html](docs/getting-started.html)** — quick-start for Python/R users
- **[examples/](examples/)** — six runnable scenario files

To regenerate the HTML docs after editing `docs/src/`:
```bash
make docs
```
```

- [ ] **Step 4: Final build and full test**

```bash
python docs/build.py
make test
```

Expected: all 25 tests pass, 14 HTML pages generated cleanly.

- [ ] **Step 5: Commit**

```bash
git add -u
git add README.md .gitignore
git commit -m "docs: cleanup old examples, update README and gitignore"
```
