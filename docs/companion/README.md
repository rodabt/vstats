# SaaS Experiment Analytics Companion

A recipe-card reference for rigorous experiment analytics. Look up a card by lifecycle phase or by the symptom you're seeing.

---

## Navigate by lifecycle phase

| Phase | Cards |
|-------|-------|
| **Planning** | [Randomization unit](01-design.md#randomization-unit) · [Baseline & time zero](01-design.md#baseline-and-time-zero) · [Metric taxonomy](01-design.md#metric-taxonomy) · [Power analysis](01-design.md#power-analysis) · [MDE to business value](01-design.md#mde-to-business-value) |
| **Running** | [SRM](02-data-quality.md#sample-ratio-mismatch-srm) · [ICC & contamination](02-data-quality.md#intraclass-correlation-icc-and-contamination) · [Novelty effects](02-data-quality.md#novelty-and-primacy-effects) · [Sequential analysis](06-readout.md#sequential-analysis-and-early-stopping) |
| **Readout** | [HTE](06-readout.md#heterogeneous-treatment-effects-hte) · [Multiple testing](06-readout.md#multiple-testing-correction) · [ITT vs per-protocol](06-readout.md#itt-vs-per-protocol) · [Mediation](06-readout.md#mediation) · [Bayesian readout](06-readout.md#bayesian-readout) |
| **Observational / Post-hoc** | [PSM](05-causal-methods.md#propensity-score-matching-psm) · [DiD](05-causal-methods.md#difference-in-differences-did) · [IV](05-causal-methods.md#instrumental-variables-iv) · [Doubly robust](05-causal-methods.md#doubly-robust-estimation-aipw) · [Target trial emulation](05-causal-methods.md#target-trial-emulation) |
| **Any phase** | [Foundations](00-foundations.md) · [Simpson's paradox](03-metric-pitfalls.md#simpsons-paradox) · [Ratio metrics](03-metric-pitfalls.md#ratio-metrics) · [CUPED](04-variance-reduction.md#cuped) |

---

## Navigate by symptom

| I'm seeing... | Go to |
|---------------|-------|
| Unequal group sizes after launch | [SRM](02-data-quality.md#sample-ratio-mismatch-srm) |
| Revenue metric is too noisy to detect an effect | [CUPED](04-variance-reduction.md#cuped) · [Winsorization](04-variance-reduction.md#winsorization) · [Delta method](04-variance-reduction.md#delta-method-for-ratio-metric-variance) |
| Significant result but the direction feels wrong | [Simpson's paradox](03-metric-pitfalls.md#simpsons-paradox) · [Selection bias](02-data-quality.md#selection-bias) · [Novelty effects](02-data-quality.md#novelty-and-primacy-effects) |
| Can't randomize — rollout already happened | [PSM](05-causal-methods.md#propensity-score-matching-psm) · [DiD](05-causal-methods.md#difference-in-differences-did) · [Target trial emulation](05-causal-methods.md#target-trial-emulation) |
| Segment results differ wildly from the average | [HTE](06-readout.md#heterogeneous-treatment-effects-hte) · [Aggregation bias](03-metric-pitfalls.md#aggregation-bias) · [ICC](02-data-quality.md#intraclass-correlation-icc-and-contamination) |
| Stakeholders question whether the result is causal | [Association vs. causation](00-foundations.md#association-vs-causation) · [Causal claims](07-communication.md#when-to-say-caused-vs-associated-with) · [ITT vs per-protocol](06-readout.md#itt-vs-per-protocol) |
| Need to stop the experiment early | [Sequential analysis](06-readout.md#sequential-analysis-and-early-stopping) |
| Many secondary metrics are all "significant" | [Multiple testing](06-readout.md#multiple-testing-correction) |
| Users in the same account are in different variants | [Randomization unit](01-design.md#randomization-unit) · [ICC](02-data-quality.md#intraclass-correlation-icc-and-contamination) |
| Metric definition looks wrong after launch | [Ratio metrics](03-metric-pitfalls.md#ratio-metrics) · [Aggregation bias](03-metric-pitfalls.md#aggregation-bias) |
| Engagement metrics for the feature look great, revenue is flat | [Mediation](06-readout.md#mediation) · [Proxy metrics](01-design.md#metric-taxonomy) |
| Top accounts at baseline look worse at readout | [Regression to the mean](03-metric-pitfalls.md#regression-to-the-mean) |

---

## Card template

Every card follows this structure:

> **One-line summary**

### When to use
### Why
### How
### Pitfalls
### Python
### SQL
### vstats (future)

---

## Relation to existing project material

- **`EXPERIMENT_CHECKLIST.md`** — pre-launch checklist; this companion provides the analytics depth behind each checklist item
- **`EXPERIMENT_TYPE_WIZARD.md`** — experiment type decision flowchart; this companion provides the how-to for each branch
- **`experiment/`** — vstats implementations of PSM, DiD, CUPED, sequential, Bayesian A/B; companion cards reference these with `vstats (future)` placeholders
