import vstats.stats
import math

fn test__bh_correction_rejects_three() {
	// BH with known result: p=[0.001,0.01,0.02,0.2,0.5], alpha=0.05
	// BH thresholds (rank*alpha/n): 0.01,0.02,0.03,0.04,0.05
	// p[0]=0.001<=0.01 ✓, p[1]=0.01<=0.02 ✓, p[2]=0.02<=0.03 ✓, rest: NO
	p_values := [0.001, 0.01, 0.02, 0.2, 0.5]
	result := stats.bh_correction(p_values, 0.05)
	assert result.n_rejected == 3
	assert result.reject[0] == true
	assert result.reject[1] == true
	assert result.reject[2] == true
	assert result.reject[3] == false
	assert result.reject[4] == false
	for adj in result.adjusted {
		assert adj >= 0.0 && adj <= 1.0
	}
}

fn test__bh_correction_rejects_none_when_all_large() {
	p_values := [0.3, 0.4, 0.5, 0.6, 0.7]
	result := stats.bh_correction(p_values, 0.05)
	assert result.n_rejected == 0
}

fn test__bh_correction_single_pvalue() {
	result := stats.bh_correction([0.03], 0.05)
	assert result.n_rejected == 1
	assert result.reject[0] == true
	assert math.abs(result.adjusted[0] - 0.03) < 1e-9
}

fn test__bonferroni_correction_rejects_one() {
	// With n=5, alpha=0.05, Bonferroni threshold = 0.01
	// p=0.005 → adj=0.025 ≤ 0.05 → reject
	// p=0.02  → adj=0.10  > 0.05  → keep
	p_values := [0.005, 0.02, 0.04, 0.1, 0.5]
	result := stats.bonferroni_correction(p_values, 0.05)
	assert result.n_rejected == 1
	assert result.reject[0] == true
	assert result.reject[1] == false
	assert math.abs(result.adjusted[0] - 0.025) < 1e-9
	assert math.abs(result.adjusted[2] - 0.2) < 1e-9
}

fn test__bonferroni_adjusted_capped_at_one() {
	p_values := [0.5, 0.5, 0.5, 0.5, 0.5]
	result := stats.bonferroni_correction(p_values, 0.05)
	for adj in result.adjusted {
		assert adj == 1.0
	}
	assert result.n_rejected == 0
}
