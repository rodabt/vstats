# Docs Reorganization — Design Spec
*Date: 2026-06-06*

## Goal

Reorganize the tutorials, examples, and docs so that:
- A **data scientist or analyst coming from Python/R** can orient immediately (no concept introductions needed — they know statistics).
- An **advanced user** can find any function's vstats equivalent and a runnable real-world scenario without digging.
- The **HTML site and `.v` example files stay in sync** permanently, not by discipline but by construction.

Out of scope: `docs/companion/` (already well-organized).

---

## 1. Directory Structure

### `docs/`

```
docs/
  src/                        ← Markdown source files (the thing you edit)
    index.md
    getting-started.md
    concepts.md
    examples.md
    modules/
      stats.md
      experiment.md
      ml.md
      hypothesis.md
      growth.md
      nn.md
      prob.md
      optim.md
      linalg.md
      utils.md
  build.py                    ← extended from companion/build_html.py
  companion/                  ← unchanged
  *.html                      ← generated output (committed for browsability)
  css/, js/                   ← unchanged
```

All existing hand-written `.html` files (`index.html`, `concepts.html`, `examples.html`, `modules/*.html`) are replaced by generated output. Their content is migrated into the corresponding `docs/src/` Markdown files.

### `examples/`

```
examples/
  rigorous-ab-readout/
    main.v
    README.md
  causal-did/
    main.v
    README.md
  churn-prediction/
    main.v
    README.md
  funnel-attribution/
    main.v
    README.md
  ratio-metric-inference/
    main.v
    README.md
  hypothesis-battery/
    main.v
    README.md
```

Existing flat `.v` files (`iris_logistic_classification.v`, `breast_cancer_classification.v`, `titanic_*.v`, `generic_types_example.v`) are consolidated into `examples/churn-prediction/` and reorganized to fit the new scenario structure. Compiled binaries (extensionless files) are added to `.gitignore`. The two `.py` files (`breast_cancer_classification.py`, `iris_logistic_classification.py`) are deleted — they are Python reimplementations of the V examples and have no place in a V library's examples directory.

---

## 2. Build Pipeline

`docs/build.py` extends `docs/companion/build_html.py` with one new capability: `<!-- include: path/to/file.v -->` tags in Markdown are replaced with fenced V code blocks at render time.

```python
def expand_includes(md_text, base_dir) -> str:
    """Replace <!-- include: foo.v --> with ```v\n<file contents>\n```"""

def render_page(src_md, output_html, template):
    """Markdown → HTML with include expansion, then apply existing CSS/JS template"""

def build_all():
    for src in glob("docs/src/**/*.md"):
        render_page(src, corresponding_html_path(src), template)
```

Build commands:
```bash
python docs/build.py    # regenerates all docs/*.html from docs/src/
make docs               # Makefile target wrapping the above
```

The `Makefile` gets a `docs` target alongside the existing `test` and `fulltest` targets.

**Dependencies:** uses Python's `markdown` library already present for the companion — no new dependencies.

**Generated HTML is committed** so users can browse without running the build. `docs/src/*.md` is the canonical source.

---

## 3. Content

### `docs/src/getting-started.md`

The "For Python/R users" landing page. No concept explanations. Opens with a translation table:

| Python / scipy / statsmodels | vstats |
|------------------------------|--------|
| `scipy.stats.ttest_ind(a, b)` | `experiment.abtest(a, b)` |
| `scipy.stats.mannwhitneyu(a, b)` | `hypothesis.mann_whitney_u_test(a, b)` |
| `statsmodels OLS + HC SE` | `experiment.ancova(...)` |
| `sklearn.linear_model.LogisticRegression` | `ml.logistic_regression(...)` |
| `scipy.stats.chi2_contingency` | `hypothesis.chi_squared_test(...)` |
| `statsmodels.stats.multitest.multipletests` | `stats.bh_correction(...)` |
| `scipy.stats.bootstrap` | `stats.bootstrap_test(...)` |

Followed by a 10–15 line complete runnable example per major domain (stats, experiment, ml, growth).

### `docs/src/modules/*.md`

Each module page keeps its existing API reference and gains a **"vs Python" callout** on every major function group. No concept introductions — those live in `concepts.md`. Links to the relevant scenario example where applicable.

### `docs/src/examples.md`

One section per scenario. Each section contains:
1. A 2-sentence "what problem this solves and why you'd reach for vstats"
2. `<!-- include: examples/<scenario>/main.v -->` (renders as a full code block in the HTML)

The `README.md` inside each scenario folder is the prose companion for GitHub browsing.

### `examples/<scenario>/main.v` — Consistent structure

Every scenario file follows this layout:
```v
// Scenario: <name>
// Demonstrates: <module> — <what makes this unique vs Python>

module main

import vstats.<module>

fn main() {
    // --- Setup ---
    // --- Core analysis ---
    // --- Interpret output ---
}
```

Output is meaningful and labeled (not walls of `println`). Every file is runnable with `v run examples/<scenario>/main.v`.

**ML examples** (`churn-prediction`) are verified for API correctness via `v build examples/churn-prediction/main.v` but not executed — runtime is too long for a test loop. All other scenario examples are verified with `v run`.

### The 6 Scenarios

| Scenario | Module | Unique angle vs Python |
|----------|--------|------------------------|
| `rigorous-ab-readout` | `experiment` | SRM + CUPED + BH correction + `null_verdict` in one pipeline |
| `causal-did` | `experiment` | DiD regression + parallel trends test — no statsmodels needed |
| `churn-prediction` | `ml` | Full train/eval pipeline with Random Forest + ROC in pure V |
| `funnel-attribution` | `growth` | Funnel drop-off + multi-touch attribution in one import |
| `ratio-metric-inference` | `stats` | Delta method + permutation bootstrap for revenue/session metrics |
| `hypothesis-battery` | `hypothesis` | Normality check → parametric/non-parametric decision in one file |

---

## 4. Navigation

### `docs/src/index.md`

Two-path entry at the top:
```
New here? → Getting Started (Python/R users)
Know what you want? → Module Reference
```

### Sidebar (all pages)

```
Getting Started
  └── For Python/R users

Examples
  ├── rigorous-ab-readout
  ├── causal-did
  ├── churn-prediction
  ├── funnel-attribution
  ├── ratio-metric-inference
  └── hypothesis-battery

Module Reference
  ├── stats
  ├── experiment
  ├── ml
  ├── hypothesis
  ├── growth
  └── prob / optim / linalg / utils / nn

Concepts
Companion Docs
```

### `README.md`

One-line update pointing to the two entry paths:
- `docs/index.html` — full browsable site
- `examples/` — runnable scenario code

---

## 5. Constraints & Non-Goals

- **No new Python dependencies** — build uses only libraries already present for the companion.
- **No framework** (MkDocs, Docusaurus, etc.) — the existing lightweight build script is sufficient and keeps the repo dependency-free.
- **`docs/companion/` is not touched.**
- **`symbol/` module** is WIP and excluded from module docs.
- **ML example execution** is skipped in CI; API correctness is verified by compile check only.
