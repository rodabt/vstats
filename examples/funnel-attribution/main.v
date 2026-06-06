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
	println('   Worst drop-off: ${drop.from_name} → ${drop.to_name}')

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
		println('   ${r.channel:-15s}  conversions=${r.conversions}  revenue=\$${r.revenue:.0f}')
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
