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

	// --- Scenario B: bimodal — users either churn fast or stay very long (no middle) ---
	// Shapiro-Wilk detects bimodal distributions as non-normal (p <= 0.05)
	b1 := [1.0, 1.0, 1.0, 1.0, 1.0, 30.0, 30.0, 30.0, 30.0, 30.0]
	b2 := [2.0, 2.0, 2.0, 2.0, 2.0, 35.0, 35.0, 35.0, 35.0, 35.0]

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
