# Examples

Seven end-to-end scenarios, each targeting a different module and showcasing
something you can't do in one call with scipy or sklearn.

All examples are runnable: `v run examples/<scenario>/main.v`
(exception: `classification-pipeline` and `titanic-survival` — verify with `v -check` only, runtime is slow)

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

## classification-pipeline

Full binary classification pipeline on the Breast Cancer Wisconsin dataset:
normalize, train logistic regression and random forest, evaluate with F1 and AUC.
(Verify with `v -check` — slow to run.)

<!-- include: examples/classification-pipeline/main.v -->

---

## titanic-survival

Three classifiers (logistic regression, Naive Bayes, random forest) trained on the
same Titanic split and compared head-to-head by accuracy, precision, recall, and F1.
(Verify with `v -check` — slow to run.)

<!-- include: examples/titanic-survival/main.v -->

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
(Mann-Whitney U) decision, applied to near-normal and bimodal scenarios.

<!-- include: examples/hypothesis-battery/main.v -->
