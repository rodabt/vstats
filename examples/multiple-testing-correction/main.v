// Scenario: Multi-Metric A/B Readout
// Demonstrates: bonferroni, holm, bh_fdr — correcting for 5 simultaneous metrics
// Python equivalent: statsmodels multipletests
module main

import vstats.experiment

fn main() {
	println('=== Multi-Metric Correction ===\n')

	metric_names := ['Revenue/user', 'Conversion rate', 'Bounce rate', 'Session length', 'Pages/session']
	p_values     := [0.003, 0.021, 0.12, 0.34, 0.87]

	println('Raw p-values:')
	for i, name in metric_names {
		mark := if p_values[i] < 0.05 { '*' } else { ' ' }
		println('  [${mark}] ${name:-20s}  p = ${p_values[i]:.3f}')
	}

	bonf := experiment.bonferroni(p_values, 0.05)
	holm := experiment.holm(p_values, 0.05)
	bh   := experiment.bh_fdr(p_values, 0.05)

	println('\nBonferroni (FWER — most conservative):   ${bonf.n_rejected} rejected')
	for i, name in metric_names {
		mark := if bonf.rejected[i] { '*' } else { ' ' }
		println('  [${mark}] ${name:-20s}  adj_p = ${bonf.adjusted_p_values[i]:.3f}')
	}

	println('\nHolm (FWER — step-down):                 ${holm.n_rejected} rejected')
	for i, name in metric_names {
		mark := if holm.rejected[i] { '*' } else { ' ' }
		println('  [${mark}] ${name:-20s}  adj_p = ${holm.adjusted_p_values[i]:.3f}')
	}

	println('\nBenjamini-Hochberg FDR (recommended):    ${bh.n_rejected} rejected')
	for i, name in metric_names {
		mark := if bh.rejected[i] { '*' } else { ' ' }
		println('  [${mark}] ${name:-20s}  adj_p = ${bh.adjusted_p_values[i]:.3f}')
	}

	println('\n* = rejected at alpha=0.05 after correction')
}
