# SaaS Experiment Analytics Companion — Design Spec

**Date:** 2026-06-02
**Status:** Approved
**Output format:** Markdown structured for eventual publication (site or PDF export)

---

## Purpose

A practical reference companion for running rigorous experiments in SaaS products. It bridges the causal inference theory from Hernán & Robins' *Causal Inference: What If* into the day-to-day language and tooling of a data science team. The primary audience is the DS practitioner (A), with the DS team as a secondary audience (B), and cross-functional experimenters (PMs, engineers) as a tertiary audience (C).

This is not a textbook. It is a codebook — a set of self-contained recipe cards that you look up when you hit a problem during an experiment. Each card is independently linkable.

---

## Source Material

- Hernán MA, Robins JM (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Existing project material: `EXPERIMENT_CHECKLIST.md`, `EXPERIMENT_TYPE_WIZARD.md`, `sequential_rollout.md`
- Existing vstats implementations: `experiment/abtest.v`, `experiment/psm.v`, `experiment/did.v`, `experiment/sequential.v`, `experiment/bayesian.v`, `experiment/sample_size.v`

Academic terminology from the book is translated into SaaS/tech language throughout. The companion does not introduce new academic jargon — it maps rigorous concepts onto the language practitioners already use.

---

## File Structure

```
docs/companion/
├── README.md                ← navigation index (lifecycle view + symptom/problem view)
├── 00-foundations.md        ← counterfactuals, identifiability, DAGs
├── 01-design.md             ← randomization unit, baseline, metric taxonomy, power
├── 02-data-quality.md       ← SRM, ICC, contamination, selection bias, novelty effects
├── 03-metric-pitfalls.md    ← Simpson's, skewness, ratio metrics, aggregation bias
├── 04-variance-reduction.md ← CUPED, stratification, winsorization, delta method
├── 05-causal-methods.md     ← PSM, DiD, IV, doubly robust, target trial emulation
├── 06-readout.md            ← HTE, multiple testing, ITT, mediation, sequential, Bayesian
└── 07-communication.md      ← business language, causal claims, null results
```

---

## Navigation Index Structure (README.md)

The `README.md` provides two cross-reference tables:

**By lifecycle phase:**

| Phase | Cards |
|-------|-------|
| Planning | Randomization unit, Baseline definition, Metric taxonomy, Power analysis, MDE |
| Running | SRM, ICC/contamination, Novelty effects, Sequential analysis |
| Readout | HTE, Multiple testing, ITT vs per-protocol, Mediation, Bayesian readout |
| Observational / Post-hoc | PSM, DiD, IV, Doubly robust, Target trial emulation |
| Any phase | Foundations, Simpson's paradox, Ratio metrics, Variance reduction |

**By symptom/problem:**

| I'm seeing... | Go to |
|---------------|-------|
| Unequal group sizes | SRM |
| Revenue metric is noisy | CUPED, Winsorization, Delta method |
| Significant result but feels wrong | Simpson's paradox, Selection bias, Novelty effects |
| Can't randomize | PSM, DiD, IV, Target trial emulation |
| Segment results differ wildly | HTE, Aggregation bias, ICC/contamination |
| Stakeholders don't believe the result | Communication, Causal claims, ITT vs per-protocol |
| Need to stop early | Sequential analysis (SPRT) |
| Multiple metrics all "significant" | Multiple testing correction |

---

## Card Template

Every recipe card follows this fixed structure:

```markdown
## Card Title

> **One-line summary** for skimmers.

### When to use
Concrete trigger tied to a lifecycle phase and symptom.
Example: "You're in Planning and need to choose whether to randomize by user or account."

### Why
The statistical or causal problem this card addresses.
3–4 sentences max. No academic jargon — use SaaS concepts.

### How
Numbered steps. Actionable, not theoretical.

### Pitfalls
Bullet list of common mistakes and misinterpretations.

### Python
Self-contained code using realistic synthetic data.
Includes a Tufte-style matplotlib chart wherever the phenomenon is visual.

### SQL
Self-contained query using common SaaS table schemas:
- `experiment_assignments(user_id, account_id, variant, assigned_at)`
- `events(user_id, event_type, event_at, properties)`
- `orders(user_id, account_id, amount, created_at)`
- `users(user_id, account_id, plan, created_at)`

### vstats (future)
> 🔲 TODO: This will be superseded by the equivalent vstats function once implemented.
```

---

## Card Inventory (~35 cards)

### `00-foundations.md` — Conceptual bedrock

| Card | One-liner |
|------|-----------|
| The counterfactual question | Every experiment asks: "what would have happened without the change?" |
| Exchangeability, positivity, consistency | The three conditions that make a causal estimate trustworthy |
| Reading a DAG in SaaS | How to draw and use causal graphs for experiment design |
| Association vs. causation | When a statistically significant result is not a causal effect |

### `01-design.md` — Planning

| Card | One-liner |
|------|-----------|
| Choosing the randomization unit | User vs. account vs. session vs. geo — and why it changes everything |
| Defining baseline and time zero | How to set the pre-experiment window and avoid leakage |
| Metric taxonomy | Primary, secondary, guardrail, and proxy metrics — and how to pick them |
| Power analysis for common metric types | Continuous, binary, and ratio metrics each need different formulas |
| MDE: from statistics to business value | Translating "minimum detectable effect" into ARR or conversion points |

### `02-data-quality.md` — Running

| Card | One-liner |
|------|-----------|
| Sample Ratio Mismatch (SRM) | When your group sizes don't match the allocation — stop, don't analyze |
| Intraclass correlation (ICC) and contamination | How much users in the same account infect each other's measurements |
| Network effects and interference | When treating one user changes another user's outcome |
| Selection bias: who ends up in your experiment | Self-selection, trigger-based exposure, and survivor filters |
| Novelty and primacy effects | Why week-1 results lie and how to detect it |
| Survivor bias in engagement metrics | Why "active users" cohorts always look better than they are |

### `03-metric-pitfalls.md` — Any phase

| Card | One-liner |
|------|-----------|
| Simpson's paradox | An aggregate effect that reverses inside every subgroup |
| Skewness and heavy tails in revenue metrics | Why a single $50k account can flip your experiment result |
| Ratio metrics: mean(a/b) ≠ mean(a)/mean(b) | The most common metric definition bug in SaaS experiments |
| Aggregation bias | Why user-level, session-level, and event-level metrics tell different stories |
| Regression to the mean | Why top-performing cohorts always look worse on re-measurement |

### `04-variance-reduction.md` — Planning + Readout

| Card | One-liner |
|------|-----------|
| CUPED: pre-experiment covariate adjustment | Use last period's metric to cut required sample size by 30–50% |
| Stratified randomization and post-stratification | Lock in balance before the experiment starts |
| Winsorization for skewed metrics | Cap outliers before computing means — with a principled threshold |
| Delta method for ratio metric variance | The correct standard error when your metric is a ratio |

### `05-causal-methods.md` — Observational / Post-hoc

| Card | One-liner |
|------|-----------|
| Propensity score matching (PSM) | Match treated and untreated units on observable confounders |
| Difference-in-differences (DiD) | Before/after comparison with a control group |
| Instrumental variables (IV) | Use randomization as a lever to estimate real-world compliance effects |
| Doubly robust estimation (AIPW) | Combines outcome model + propensity model; one can be wrong |
| Target trial emulation | Turn observational log data into a virtual RCT |

### `06-readout.md` — Readout

| Card | One-liner |
|------|-----------|
| Heterogeneous treatment effects (HTE) | When the average effect hides opposite effects in subgroups |
| Multiple testing correction | Bonferroni and Benjamini-Hochberg for secondary metric lists |
| ITT vs. per-protocol analysis | Intent-to-treat vs. actual exposure — and when each is right |
| Mediation: mechanism vs. total effect | Is the feature working through the path you designed? |
| Sequential analysis and early stopping | Pre-registered stopping rules using SPRT |
| Bayesian readout | Probability of being better, expected loss, credible intervals |

### `07-communication.md` — Any phase

| Card | One-liner |
|------|-----------|
| Translating CIs into business language | "We are 95% confident the effect is between X and Y revenue" |
| When to say "caused" vs. "associated with" | The conditions that license a causal claim |
| Writing the experiment narrative | Hypothesis → result → recommendation in 200 words |
| Communicating null results | How to ship a "no effect" finding without losing credibility |

---

## Code Conventions

**Python examples:**
- Generate synthetic data inline (no external dataset dependencies)
- Use `pandas`, `numpy`, `scipy.stats`, `statsmodels` — standard DS stack
- Charts use `matplotlib` with Tufte-style defaults: minimal grid, no top/right spines, direct labels, muted palette
- Each code block is fully self-contained (imports included)

**SQL examples:**
- Written for standard SQL (compatible with BigQuery, Snowflake, Redshift with minor dialect changes)
- Use the four canonical SaaS tables defined in the card template
- Include comments explaining each CTE

**Tufte chart defaults (applied to all figures):**
```python
# Applied at the top of each chart block
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'font.family': 'sans-serif',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})
```

---

## vstats Integration Placeholder

Every card ends with a `### vstats (future)` block. This marks:
- Which vstats module would own the implementation (`experiment/`, `hypothesis/`, `stats/`, etc.)
- The anticipated function signature (even if not yet implemented)

Example:
```
> 🔲 TODO: `experiment.cuped(y, y_pre)` — variance-reduced treatment effect estimator.
> Tracked in vstats roadmap.
```

This placeholder is the integration contract between the companion and the codebase.

---

## Out of Scope

- No web UI or interactive elements (this is Markdown for eventual publication)
- No new vstats code written during companion authoring — placeholders only
- No coverage of techniques with no practical SaaS analog (e.g., time-varying dose regimens from the book's Part III are omitted; DiD and target trial emulation cover the relevant intuitions)
- No duplication of content already in `EXPERIMENT_CHECKLIST.md` or `EXPERIMENT_TYPE_WIZARD.md` — companion links to those instead

---

## Success Criteria

1. A DS team member can look up any card during an experiment and get a working code snippet in under 2 minutes
2. A PM can read the "When to use" and "Pitfalls" sections of any card and understand the key risk without reading the rest
3. Every causal claim in the companion is grounded in an identifiability condition from the book
4. Every chart illustrates the phenomenon rather than decorating the page
