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

- [Examples](examples.html) — end-to-end scenarios
- [Module Reference](modules/stats.html) — full API
