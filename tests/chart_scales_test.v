import chart

fn test__linear_scale_maps_domain_to_range() {
	s := chart.LinearScale{
		domain_min: 0.0
		domain_max: 10.0
		range_min:  0.0
		range_max:  100.0
	}
	assert s.map(0.0) == 0.0
	assert s.map(5.0) == 50.0
	assert s.map(10.0) == 100.0
}

fn test__linear_scale_handles_inverted_range_for_y_flip() {
	s := chart.LinearScale{
		domain_min: 0.0
		domain_max: 10.0
		range_min:  400.0
		range_max:  0.0
	}
	assert s.map(0.0) == 400.0
	assert s.map(10.0) == 0.0
	assert s.map(5.0) == 200.0
}

fn test__linear_scale_degenerate_domain_returns_range_midpoint() {
	s := chart.LinearScale{
		domain_min: 5.0
		domain_max: 5.0
		range_min:  0.0
		range_max:  100.0
	}
	assert s.map(5.0) == 50.0
}

fn test__nice_ticks_produces_round_steps() {
	assert chart.nice_ticks(0.0, 100.0, 5) == [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
}

fn test__nice_ticks_equal_min_max() {
	assert chart.nice_ticks(7.0, 7.0, 5) == [7.0]
}

fn test__fmt_tick_integers_have_no_decimals() {
	assert chart.fmt_tick(40.0) == '40'
	assert chart.fmt_tick(0.0) == '0'
}

fn test__fmt_tick_fractions_get_two_decimals() {
	assert chart.fmt_tick(2.5) == '2.50'
}
