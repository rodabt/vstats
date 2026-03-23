module main

import growth

fn test_first_touch_attributes() {
	customer_channels := ['Google', 'Facebook', 'Google', 'Email']
	conversions := [true, true, true, true]
	revenue := [100.0, 200.0, 150.0, 175.0]

	results := growth.first_touch_attributes(customer_channels, conversions, revenue)

	// Google gets 2 conversions (100 + 150), Facebook 1 (200), Email 1 (175)
	assert results.len >= 3
}

fn test_last_touch_attributes() {
	customer_channels := ['Google', 'Facebook', 'Organic', 'Email']
	conversions := [true, true, true, true]
	revenue := [100.0, 200.0, 150.0, 175.0]

	results := growth.last_touch_attributes(customer_channels, conversions, revenue)

	// Last touch: Google, Facebook, Organic, Email (in order)
	assert results.len == 4
}

fn test_linear_attributes() {
	customer_touchpoints := [
		['Google', 'Facebook'],
		['Email', 'Google', 'Organic'],
	]
	conversions := [true, true]
	revenue := [100.0, 200.0]

	results := growth.linear_attributes(customer_touchpoints, conversions, revenue)

	// Google: 50 + 66.67 = 116.67
	// Facebook: 50
	// Email: 66.67
	// Organic: 66.67
	assert results.len == 4
}

fn test_time_decay_attributes() {
	customer_touchpoints := [
		['Google', 'Facebook'],
		['Email', 'Google'],
	]
	touchpoint_days := [
		[7, 1],
		[14, 1],
	]
	conversions := [true, true]
	revenue := [100.0, 200.0]

	results := growth.time_decay_attributes(customer_touchpoints, touchpoint_days, conversions, revenue, 7.0)

	assert results.len >= 2
}

fn test_position_based_attributes() {
	customer_touchpoints := [
		['Google', 'Facebook', 'Email'],
		['Organic', 'Google'],
	]
	conversions := [true, true]
	revenue := [300.0, 200.0]

	results := growth.position_based_attributes(customer_touchpoints, conversions, revenue)

	// First customer: Google=120 (40%), Facebook=60 (20%), Email=120 (40%)
	// Second customer: Organic=80 (40%), Google=120 (60% actual = 40% first + last overlap)
	assert results.len >= 3
}

fn test_channel_roi() {
	results := [
		growth.AttributionResult{
			channel: 'Google'
			conversions: 10
			revenue: 1000.0
			attribution_score: 0.5
		},
		growth.AttributionResult{
			channel: 'Facebook'
			conversions: 5
			revenue: 500.0
			attribution_score: 0.25
		},
	]

	channel_costs := {
		'Google':   200.0
		'Facebook': 100.0
	}

	roi := growth.channel_roi(results, channel_costs)

	assert roi['Google'] == 4.0   // (1000-200)/200
	assert roi['Facebook'] == 4.0  // (500-100)/100
}

fn test_roas() {
	// Google: $1000 revenue, $250 spend -> 4x ROAS
	result := growth.roas(1000.0, 250.0)
	assert result == 4.0

	// Zero ad spend
	result2 := growth.roas(1000.0, 0.0)
	assert result2 == 0.0
}

fn test_blended_roas() {
	result := growth.blended_roas(10000.0, 2500.0)
	assert result == 4.0
}

fn test_optimal_channel_mix() {
	channel_performance := {
		'Google':   0.10  // 10% conversion rate
		'Facebook': 0.05  // 5% conversion rate
		'Email':    0.08  // 8% conversion rate
	}

	total_budget := 1000.0
	allocation := growth.optimal_channel_mix(channel_performance, total_budget)

	// Total: 0.10 + 0.05 + 0.08 = 0.23
	// Google: 0.10/0.23 * 1000 = 434.78
	// Facebook: 0.05/0.23 * 1000 = 217.39
	// Email: 0.08/0.23 * 1000 = 347.83
	assert allocation['Google'] > 0.0
	assert allocation['Facebook'] > 0.0
	assert allocation['Email'] > 0.0
}
