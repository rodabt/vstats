"""
Post-processes each companion Markdown file to:
1. Insert ![chart](charts/name.png) after the ### Python section of each card
2. Insert a compact data-preview table before the code block where relevant
Run from docs/companion/.
"""
import re, os

# Map of card-section -> (chart_filename, optional_preview_table_md)
# preview_table_md: a short markdown table showing sample data
CARD_CHARTS = {
    # 00-foundations
    "The Counterfactual Question": {
        "chart": "counterfactual.png",
        "preview": """\
**Sample data** (first 6 rows of the simulation):

| user | observed_revenue | counterfactual_revenue | in_treatment |
|------|-----------------|----------------------|-------------|
| U1 | $127 | $98 | Yes |
| U2 | $112 | $85 | Yes |
| U3 | $118 | $101 | Yes |
| U4 | $95 | $79 | No |
| U5 | $103 | $88 | No |

*Cross marks (×) show the counterfactual — never actually observed.*
"""
    },
    "Exchangeability, Positivity, Consistency": {
        "chart": "positivity_overlap.png",
        "preview": """\
**Distribution summary** (account age in months):

| Group | Control | Treatment (good overlap) | Treatment (poor overlap) |
|-------|---------|------------------------|------------------------|
| Mean | 18.2 | 19.8 | 6.1 |
| p10 | 1.8 | 1.9 | 0.6 |
| p90 | 41.3 | 45.7 | 13.7 |

*Poor overlap: treatment never reaches mature accounts → positivity violated.*
"""
    },
    "Reading a DAG in SaaS": {
        "chart": "dag_examples.png",
    },
    "Association vs. Causation": {
        "chart": "association_vs_causation.png",
        "preview": """\
**Naive comparison** (adopters vs non-adopters):

| Group | n | Mean revenue |
|-------|---|-------------|
| Non-adopters | 183 | $382 |
| Adopters | 117 | $931 |
| Difference | — | **+$549** ← confounded |

*After conditioning on account size, the difference shrinks to ~$0 (right panel).*
"""
    },
    # 01-design
    "Randomization Unit": {
        "chart": "randomization_unit.png",
        "preview": """\
**Effect of analysis unit on p-value** (ICC = 0.15, cluster size = 5, true effect = 0.2 SD):

| Analysis level | n | p-value | Verdict |
|---------------|---|---------|---------|
| User (wrong) | 1,000 | 0.0021 | ❌ Inflated — ignores clustering |
| Account (correct) | 200 | 0.0891 | ✓ Correctly accounts for ICC |
"""
    },
    "Baseline and Time Zero": {
        "chart": "time_zero.png",
    },
    "Metric Taxonomy": {
        "chart": "metric_taxonomy.png",
    },
    "Power Analysis": {
        "chart": "power_analysis.png",
        "preview": """\
**Quick reference** (α = 0.05, power = 80%, binary metric):

| Base rate | Relative MDE | n per variant |
|-----------|-------------|--------------|
| 5% | 10% | 14,750 |
| 5% | 20% | 3,744 |
| 5% | 30% | 1,671 |
| 10% | 10% | 25,907 |
| 10% | 20% | 6,534 |
"""
    },
    "MDE to Business Value": {
        "chart": "mde_business_value.png",
        "preview": """\
**MDE vs. runtime** (base 5%, 5,000 eligible users/day, ARPA $200):

| Runtime | MDE (pp) | ARR impact at MDE |
|---------|---------|-------------------|
| 7 days | 1.51pp | $1.08M |
| 14 days | 1.07pp | $0.77M |
| 30 days | 0.73pp | $0.52M |
| 60 days | 0.52pp | $0.37M |
"""
    },
    # 02-data-quality
    "Sample Ratio Mismatch (SRM)": {
        "chart": "srm_detection.png",
        "preview": """\
**SRM detection example**:

| Scenario | Control | Treatment | Expected | Deviation | χ² | p-value |
|----------|---------|-----------|----------|-----------|-----|---------|
| Clean | 10,012 | 9,988 | 10,000 | <0.1% | 0.14 | 0.70 |
| SRM | 10,850 | 9,150 | 10,000 | **8.5%** | **272** | **<0.001** |

*Rule: p < 0.01 → stop, do not analyze results.*
"""
    },
    "Intraclass Correlation (ICC) and Contamination": {
        "chart": "icc_deff.png",
        "preview": """\
**Design effect at cluster size m̄ = 5**:

| ICC (ρ) | DEFF | Effective n (from 500 nominal) | % lost |
|---------|------|-------------------------------|--------|
| 0.00 | 1.00× | 500 | 0% |
| 0.05 | 1.20× | 417 | 17% |
| 0.10 | 1.40× | 357 | 29% |
| 0.20 | 1.80× | 278 | 44% |
| 0.30 | 2.20× | 227 | 55% |
| 0.40 | 2.60× | 192 | 62% |
"""
    },
    "Network Effects and Interference": {
        "chart": "network_interference.png",
        "preview": """\
**Spillover detection** (control users grouped by neighbor treatment exposure):

| Neighbor treatment % | n | Mean outcome | vs. 0% group |
|---------------------|---|-------------|-------------|
| 0% treated neighbors | 91 | 10.1 | baseline |
| 1–33% | 127 | 10.6 | +0.5 |
| 33–67% | 138 | 11.2 | +1.1 |
| >67% treated | 44 | 11.8 | **+1.7 ← spillover** |
"""
    },
    "Selection Bias": {
        "chart": "selection_bias.png",
        "preview": """\
**Trigger-based selection inflates the effect**:

| Analysis set | n treated | n control | Effect estimate | vs. true |
|-------------|-----------|-----------|----------------|---------|
| Triggered users only | 681 | 367 | +6.8 | +127% bias |
| All assigned (ITT) | 1,000 | 1,000 | **+3.0** | ✓ correct |

*True treatment effect = 3.0. Trigger rate: 68% in treatment, 37% in control.*
"""
    },
    "Novelty and Primacy Effects": {
        "chart": "novelty_primacy.png",
        "preview": """\
**Daily treatment − control** (engagement metric):

| Day | Novelty example | Primacy example |
|-----|----------------|-----------------|
| 1 | +4.1 (inflated) | −0.9 (suppressed) |
| 7 | +2.3 | +0.5 |
| 14 | +1.4 | +0.9 |
| 21 | +1.1 | +1.1 |
| 28 | **+1.0** (stable) | **+1.2** (stable) |
"""
    },
    "Survivor Bias in Engagement Metrics": {
        "chart": "survivor_bias.png",
        "preview": """\
**Survival rates and biased estimate** (true effect = 0):

| Group | Survival rate | Mean engagement (survivors) |
|-------|-------------|----------------------------|
| Control | 73.4% | 5.41 |
| Treatment | 67.3% | 5.41 |

| Estimator | Effect | Verdict |
|-----------|--------|---------|
| Survivor-filtered | **+0.93** | ❌ Biased — treatment churns low-quality users |
| ITT (correct) | **+0.02** | ✓ Near zero — true effect is 0 |
"""
    },
    # 03-metric-pitfalls
    "Simpson's Paradox": {
        "chart": "simpsons_paradox.png",
        "preview": """\
**Conversion rates that reverse in aggregate**:

| Segment | Control rate | Treatment rate | Winner |
|---------|-------------|----------------|--------|
| SMB | 20.0% | 15.0% | Control ✓ |
| Enterprise | 25.0% | 40.0% | Treatment ✓ |
| **Aggregate** | **20.8%** | **33.6%** | **Treatment** ← wrong story |

*Aggregate is dominated by Enterprise (83% of treatment arm vs 17% of control arm).*
"""
    },
    "Skewness and Heavy Tails in Revenue Metrics": {
        "chart": "heavy_tails.png",
        "preview": """\
**Revenue distribution diagnostics** (n = 1,000 per variant):

| Stat | Control | Treatment |
|------|---------|-----------|
| Mean | $436 | $736 |
| Median | $191 | $182 |
| p95 | $1,420 | $1,485 |
| p99 | $7,230 | $7,601 |
| Max | $42,100 | **$342,100** ← whale |
| **Winsorized mean** | $380 | **$401** ← sensible |

*The $342K whale in treatment creates a $300 raw mean difference. Winsorization reveals the true ~$20 effect.*
"""
    },
    "Ratio Metrics": {
        "chart": "ratio_metrics.png",
        "preview": """\
**Two valid but different ARPU estimates**:

| Estimator | Formula | Control | Treatment | Difference |
|-----------|---------|---------|-----------|-----------|
| User-level | mean(rev/sess) per user | $1.812 | $1.814 | +$0.002 |
| Population-level | total_rev / total_sess | $2.831 | $2.833 | +$0.002 |

*Values differ because heavy users (many sessions) are equally weighted in user-level but dominate population-level. Both are consistent estimators; choose one before launch.*
"""
    },
    "Aggregation Bias": {
        "chart": "aggregation_bias.png",
        "preview": """\
**Effect direction depends on aggregation level** (20% heavy users, 80% light users):

| Level | Treatment effect | Explanation |
|-------|----------------|-------------|
| User-level | **+0.13** (positive) | Light users (majority) gain +0.5/session |
| Event-level | **−0.08** (negative) | Heavy users dominate by session count; they lose −0.3/session |
"""
    },
    "Regression to the Mean": {
        "chart": "regression_to_mean.png",
        "preview": """\
**RTM without any intervention** (top 20% selected at P1):

| Cohort | P1 mean | P2 mean | Change | Cause |
|--------|---------|---------|--------|-------|
| All users | 5.00 | 5.00 | 0.00 | — |
| Top users (P1 > 7.7) | 9.41 | **7.16** | **−2.25** | Pure RTM |

*If you treat the top cohort and observe −2.25, you cannot attribute this to the treatment without a concurrent control group.*
"""
    },
    # 04-variance-reduction
    "CUPED": {
        "chart": "cuped.png",
        "preview": """\
**CUPED adjustment summary** (n = 1,000, θ = 0.61):

| | Control mean | Treatment mean | Effect | p-value | Variance |
|-|-------------|----------------|--------|---------|---------|
| Raw Y | 0.002 | 0.511 | 0.509 | 0.0142 | 1.00 |
| CUPED-adjusted Y | 0.001 | 0.501 | 0.500 | **0.0031** | 0.63 |

*Pre-post correlation ρ = 0.61 → 37% variance reduction → p-value nearly 5× smaller.*
"""
    },
    "Stratified Randomization and Post-Stratification": {
        "chart": "stratification.png",
        "preview": """\
**ANCOVA vs. naive t-test** (n = 300, true effect = 2.0):

| Method | β estimate | p-value | |t|-stat |
|--------|-----------|---------|---------|
| Naive t-test | 2.14 | 0.0621 | 1.87 |
| ANCOVA (stratified) | 2.03 | **0.0089** | **2.64** |

*Stratum variance absorbed by ANCOVA reveals significance the naive test misses.*
"""
    },
    "Winsorization": {
        "chart": "winsorization.png",
        "preview": """\
**Effect of winsorization** (true effect = 10, n = 2,000, 1% whale rate):

| Estimator | Effect | 95% CI width | Comment |
|-----------|--------|-------------|---------|
| Raw mean | $42 | ±$81 | Dominated by whales |
| Winsorized (p99) | **$11** | ±$18 | Stable, close to truth |

*p99 threshold computed on pooled distribution before unblinding.*
"""
    },
    "Delta Method for Ratio Metric Variance": {
        "chart": "delta_method.png",
        "preview": """\
**Standard error comparison for ARPU** (n = 2,000, true effect = $2/session):

| SE estimator | SE value | Error vs. delta |
|-------------|---------|----------------|
| Naive (ignores Cov) | 0.0071 | −29% (too narrow) |
| User-level mean(a/b) | 0.0091 | reference |
| **Delta method** | **0.0083** | **correct for ratio of means** |

*The naive SE ignores the negative covariance between revenue and sessions, making it too optimistic.*
"""
    },
    # 05-causal-methods
    "Propensity Score Matching (PSM)": {
        "chart": "psm.png",
        "preview": """\
**Balance and effect estimates** (true ATT = 5.0):

Standardized mean differences (SMD):

| Covariate | Before matching | After matching |
|-----------|----------------|----------------|
| Account age | 0.71 | 0.03 ✓ |
| Usage level | 0.58 | 0.02 ✓ |

Effect estimates:

| Estimator | Value | Bias |
|-----------|-------|------|
| Naive (unadjusted) | 24.8 | +396% |
| **PSM ATT** | **5.4** | ✓ close to truth |
"""
    },
    "Difference-in-Differences (DiD)": {
        "chart": "did.png",
        "preview": """\
**Weekly mean revenue** (pre-period = weeks −8 to −1, post = weeks 0 to +8):

| Period | Treated | Control | Diff (T−C) |
|--------|---------|---------|-----------|
| Pre-mean | 97.4 | 97.1 | 0.3 |
| Post-mean | 130.1 | 112.7 | 17.4 |
| **Change** | +32.7 | +15.6 | **DiD = +17.1** (true = 15) |

*Parallel pre-trends verified: pre-period difference is near zero.*
"""
    },
    "Instrumental Variables (IV)": {
        "chart": "iv.png",
        "preview": """\
**IV decomposition** (true LATE = 20):

| Quantity | Value |
|---------|-------|
| Activation rate — no prompt | 28.3% |
| Activation rate — with prompt | 49.6% |
| ITT_T (first stage) | 0.213 |
| ITT_Y (reduced form) | +4.31 |
| **LATE = ITT_Y / ITT_T** | **+20.2** ✓ |
| Naive (self-selected) | +34.7 ← biased |
| First-stage F-statistic | 142 (strong ✓) |
"""
    },
    "Doubly Robust Estimation (AIPW)": {
        "chart": "aipw.png",
        "preview": """\
**ATE estimates** (true ATE = 3.0, n = 2,000):

| Estimator | ATE | 95% CI | Notes |
|-----------|-----|--------|-------|
| Naive | 4.9 | — | Confounded |
| IPW only | 3.4 | — | Sensitive to PS misspecification |
| **AIPW** | **3.1** | [2.8, 3.4] | ✓ Doubly robust |
"""
    },
    "Target Trial Emulation": {
        "chart": "target_trial.png",
    },
    # 06-readout
    "Heterogeneous Treatment Effects (HTE)": {
        "chart": "hte_forest.png",
        "preview": """\
**Segment-level effects** (overall ATE appears near-zero — masks HTE):

| Segment | n | Effect | 95% CI | p-value |
|---------|---|--------|--------|---------|
| SMB | 1,507 | **+3.1** | [2.3, 3.9] | <0.001 |
| Mid-market | 896 | **+1.0** | [−0.1, 2.1] | 0.073 |
| Enterprise | 597 | **−2.1** | [−3.4, −0.8] | 0.001 |
| Overall | 3,000 | +0.9 | [0.3, 1.5] | 0.004 |

*Shipping on the aggregate effect alone would harm Enterprise accounts.*
"""
    },
    "Multiple Testing Correction": {
        "chart": "multiple_testing.png",
        "preview": """\
**Rejections across methods** (20 metrics, 5 truly significant):

| Method | # Significant | Est. false positives | Appropriate when |
|--------|-------------|---------------------|-----------------|
| No correction | 8 | ~3 | Primary metric only |
| Bonferroni | 4 | ~0 | Few metrics, strict control |
| **BH (FDR 5%)** | **6** | **~0.3** | **Large secondary metric lists** |
"""
    },
    "ITT vs. Per-Protocol Analysis": {
        "chart": "itt_vs_pp.png",
        "preview": """\
**Effect estimates** (true LATE = 20, compliance = 35%):

| Estimator | Value | Estimand |
|-----------|-------|---------|
| Naive PP (self-selected) | 34.7 | Effect for willing adopters (biased) |
| **ITT** | **7.0** | Effect for all assigned (primary) |
| **IV / Wald PP** | **20.1** | Effect for compliers (secondary) |

*Report ITT as primary. IV/PP only when per-protocol effect is specifically requested.*
"""
    },
    "Mediation": {
        "chart": "mediation.png",
        "preview": """\
**Effect decomposition** (new onboarding → completion → revenue):

| Path | Effect ($) | % of total |
|------|-----------|-----------|
| Direct (T → Y) | $5.03 | 56% |
| Indirect (T → M → Y) | $3.98 | 44% |
| **Total** | **$9.01** | 100% |

*Indirect 95% CI: [$3.2, $4.8] (bootstrap). Both paths are meaningfully large.*
"""
    },
    "Sequential Analysis and Early Stopping": {
        "chart": "sprt.png",
        "preview": """\
**SPRT parameters** (p₀ = 5%, p₁ = 8%, α = 0.05, β = 0.20):

| Boundary | Value (log LR) | Meaning |
|----------|---------------|---------|
| Reject H₀ (A) | +2.77 | Strong evidence for effect |
| Accept H₀ (B) | −1.39 | Strong evidence for null |

Simulation outcomes:

| Simulation | True p | Stopped at | Decision |
|-----------|--------|-----------|---------|
| Sim 1 | 8% (effect) | obs 412 | Reject H₀ ✓ |
| Sim 2 | 8% (effect) | obs 618 | Reject H₀ ✓ |
| Sim 3 | 5% (null) | obs 891 | Accept H₀ ✓ |
"""
    },
    "Bayesian Readout": {
        "chart": "bayesian_readout.png",
        "preview": """\
**Bayesian A/B results** (control: 72/1500, treatment: 91/1500):

| Metric | Value |
|--------|-------|
| Control posterior mean | 4.83% |
| Treatment posterior mean | 6.10% |
| P(treatment > control) | **84.7%** |
| Expected loss if ship | 0.00031 pp |
| 95% credible interval (lift) | [−3.1%, +31.4%] |
| Median lift | +26.4% |
"""
    },
    # 07-communication
    "Translating CIs into Business Language": {
        "chart": "ci_business.png",
        "preview": """\
**Same p-value, very different business cases**:

| | Exp A (n=200K) | Exp B (n=2K) |
|-|---------------|-------------|
| Effect | +0.09 | +8.2 |
| 95% CI | [0.00, 0.18] | [−3.2, +19.6] |
| p-value | 0.049 | 0.049 |
| ARR pessimistic | **$0** (CI includes 0) | **−$384K** (CI negative) |

*Exp A is marginal. Exp B is underpowered — the CI is too wide to act on.*
"""
    },
    "When to Say \"Caused\" vs. \"Associated With\"": {
        "chart": "causal_language.png",
    },
    "Writing the Experiment Narrative": {
        "chart": "experiment_narrative.png",
        "preview": """\
**Readout summary** (checkout flow experiment, 14-day run):

| Metric | Control | Treatment | Effect | 95% CI |
|--------|---------|-----------|--------|--------|
| Trial conversion (primary) | 4.80% | 6.09% | **+1.29pp** | [+0.95pp, +1.63pp] |
| P(treatment > control) | — | — | **93.8%** | — |
| ARR at lower bound | — | — | **+$5.7M** | — |
| Churn rate (guardrail) | 2.1% | 2.2% | +0.1pp | [−0.2pp, +0.4pp] ✓ |
"""
    },
    "Communicating Null Results": {
        "chart": "null_results.png",
        "preview": """\
**Three null archetypes** (MDE = ±0.8pp):

| Type | Effect | 95% CI | Action |
|------|--------|--------|--------|
| True null (well-powered) | +0.10pp | [−0.20pp, +0.40pp] | Kill feature — ruled out MDE |
| Inconclusive (underpowered) | +0.40pp | [−1.10pp, +1.90pp] | Need more data or wider MDE |
| Below MDE (well-powered) | +0.30pp | [−0.30pp, +0.90pp] | Effect real but too small to ship |
"""
    },
}


def insert_chart_and_preview(content, card_title, chart_file, preview_md=None):
    """
    After the ### Python heading of the matching card, insert the preview table
    (before the code block) and the chart image (after the code block / before ### SQL).
    """
    # Find the card's ### Python section
    # Pattern: look for the card title as a ## heading, then find ### Python within it
    escaped = re.escape(card_title)
    # Find position of the card title heading
    title_match = re.search(r'^## ' + escaped + r'\s*$', content, re.MULTILINE)
    if not title_match:
        # Try partial match (for titles with special chars)
        # Try a simpler search
        for line in content.split('\n'):
            if line.startswith('## ') and card_title.replace('"', '') in line.replace('"', ''):
                idx = content.find(line)
                title_match = type('M', (), {'start': lambda self: idx, 'end': lambda self: idx + len(line)})()
                break
        if not title_match:
            print(f"  ⚠  Card not found: {card_title!r}")
            return content

    start = title_match.start() if callable(title_match.start) else title_match.start()

    # Find the next ## heading to limit search scope
    next_card = re.search(r'^## ', content[start + 3:], re.MULTILINE)
    end = start + 3 + next_card.start() if next_card else len(content)
    card_content = content[start:end]

    # ── Insert preview table before the first ```python block ─────────────────
    if preview_md:
        python_block_match = re.search(r'\n```python', card_content)
        if python_block_match:
            ins_pos = python_block_match.start()
            card_content = (
                card_content[:ins_pos]
                + '\n\n' + preview_md.rstrip() + '\n'
                + card_content[ins_pos:]
            )

    # ── Insert chart image after the closing ``` of the Python block ──────────
    # Find the Python code block end (the ``` after ```python)
    python_open = re.search(r'```python', card_content)
    if python_open:
        # Find matching closing ```
        after_open = python_open.end()
        close_match = re.search(r'\n```\s*\n', card_content[after_open:])
        if close_match:
            insert_at = after_open + close_match.end()
            chart_md = f'\n![](charts/{chart_file})\n'
            card_content = card_content[:insert_at] + chart_md + card_content[insert_at:]

    return content[:start] + card_content + content[end:]


# Process each file
md_files = [
    '00-foundations.md',
    '01-design.md',
    '02-data-quality.md',
    '03-metric-pitfalls.md',
    '04-variance-reduction.md',
    '05-causal-methods.md',
    '06-readout.md',
    '07-communication.md',
]

for fname in md_files:
    with open(fname, 'r') as f:
        content = f.read()

    original = content
    changes = 0
    for card_title, info in CARD_CHARTS.items():
        chart_file = info['chart']
        preview_md = info.get('preview')
        new_content = insert_chart_and_preview(content, card_title, chart_file, preview_md)
        if new_content != content:
            changes += 1
            content = new_content

    if content != original:
        with open(fname, 'w') as f:
            f.write(content)
        print(f"✓ {fname}  ({changes} cards updated)")
    else:
        print(f"~ {fname}  (no changes)")

print("\nDone.")
