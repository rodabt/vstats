module main

import growth
import math

fn test_create_funnel() {
	// Create a simple 3-stage funnel
	// Visit -> Signup -> Purchase
	// 1000 -> 200 -> 50
	stage_names := ['Visit', 'Signup', 'Purchase']
	stage_users := [1000, 200, 50]

	funnel := growth.create_funnel(stage_names, stage_users)

	assert funnel.stages.len == 3
	assert funnel.stages[0].users == 1000
	assert funnel.stages[1].users == 200
	assert funnel.stages[2].users == 50
	assert funnel.total_conversion == 0.05 // 5% overall conversion
}

fn test_stage_conversion_rate() {
	// Test basic conversion rate
	// 200 out of 1000 -> 20%
	result := growth.stage_conversion_rate(1000, 200)
	assert result == 0.20

	// Zero users in
	result2 := growth.stage_conversion_rate(0, 200)
	assert result2 == 0.0

	// Perfect conversion
	result3 := growth.stage_conversion_rate(100, 100)
	assert result3 == 1.0
}

fn test_stage_drop_off_rate() {
	// Test basic drop-off rate
	// 800 drop out of 1000 -> 80%
	result := growth.stage_drop_off_rate(1000, 200)
	assert result == 0.80

	// Zero users in
	result2 := growth.stage_drop_off_rate(0, 200)
	assert result2 == 0.0

	// No drop-off
	result3 := growth.stage_drop_off_rate(100, 100)
	assert result3 == 0.0
}

fn test_get_conversions() {
	stage_names := ['Visit', 'Signup', 'Purchase']
	stage_users := [1000, 200, 50]

	funnel := growth.create_funnel(stage_names, stage_users)
	conversions := funnel.get_conversions()

	assert conversions.len == 2
	assert conversions[0].from_name == 'Visit'
	assert conversions[0].to_name == 'Signup'
	assert conversions[0].rate == 0.20
	assert conversions[0].drop_off_rate == 0.80
}

fn test_highest_drop_off() {
	stage_names := ['Visit', 'Signup', 'Purchase', 'Activate']
	stage_users := [1000, 500, 100, 80]

	funnel := growth.create_funnel(stage_names, stage_users)
	highest := funnel.highest_drop_off()

	// Visit to Signup: 50% drop-off
	// Signup to Purchase: 80% drop-off (highest)
	// Purchase to Activate: 20% drop-off
	assert highest.from_name == 'Signup'
	assert highest.to_name == 'Purchase'
	assert highest.drop_off_rate == 0.80
}

fn test_lowest_drop_off() {
	stage_names := ['Visit', 'Signup', 'Purchase', 'Activate']
	stage_users := [1000, 500, 100, 80]

	funnel := growth.create_funnel(stage_names, stage_users)
	lowest := funnel.lowest_drop_off()

	// Visit to Signup: 50% drop-off
	// Signup to Purchase: 80% drop-off
	// Purchase to Activate: 20% drop-off (lowest)
	assert lowest.from_name == 'Purchase'
	assert lowest.to_name == 'Activate'
	assert math.abs(lowest.drop_off_rate - 0.20) < 0.001
}

fn test_funnel_value() {
	stage_values := [1000.0, 500.0, 250.0]
	stage_users := [1000, 500, 250]

	values := growth.funnel_value(stage_values, stage_users)

	assert values[0] == 1.0  // $1 per visitor
	assert values[1] == 1.0  // $1 per signup
	assert values[2] == 1.0  // $1 per purchaser
}

fn test_expected_revenue() {
	stage_names := ['Visit', 'Signup', 'Purchase']
	stage_users := [1000, 200, 50]
	stage_revenues := [0.0, 0.0, 100.0]

	funnel := growth.create_funnel(stage_names, stage_users)
	revenue := growth.expected_revenue(funnel, stage_revenues)

	// Expected = (1.0 × $0) + (0.2 × $0) + (0.25 × $100) = $25
	assert revenue == 25.0
}

fn test_ab_test_funnel() {
	funnel_a := growth.create_funnel(['Visit', 'Purchase'], [1000, 50])
	funnel_b := growth.create_funnel(['Visit', 'Purchase'], [1000, 75])

	// Funnel B has better conversion (7.5% vs 5%)
	result := growth.ab_test_funnel(funnel_a, funnel_b)
	assert result == 1
}

fn test_segment_funnel() {
	segment_data := {
		'desktop':   [1000, 400, 100]
		'mobile':    [1000, 200, 40]
	}

	segment_funnels := growth.segment_funnel(segment_data)

	assert segment_funnels['desktop'].total_conversion == 0.10
	assert segment_funnels['mobile'].total_conversion == 0.04
}

fn test_funnel_leakage() {
	stage_names := ['Visit', 'Signup', 'Purchase']
	stage_users := [1000, 200, 50]

	funnel := growth.create_funnel(stage_names, stage_users)
	leakage := growth.funnel_leakage(funnel)

	assert leakage.len == 2
	assert leakage[0] == 800  // Lost from Visit to Signup
	assert leakage[1] == 150  // Lost from Signup to Purchase
}

fn test_projected_conversions() {
	stage_names := ['Visit', 'Signup', 'Purchase']
	stage_users := [1000, 200, 50]

	funnel := growth.create_funnel(stage_names, stage_users)
	projections := growth.projected_conversions(funnel, 500)

	// Projections should have values
	assert projections.len >= 2
}
