import chart

fn test__grouped_bar_emits_one_rect_per_segment() {
	svg := chart.new(width: 500, height: 300)
		.grouped_bar([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]], labels: ['A', 'B', 'C'])
		.render()
	assert svg.count('<rect') >= 7 // background + 3 groups * 2 segments
	assert svg.contains('>A<')
	assert svg.contains('>B<')
	assert svg.contains('>C<')
}

fn test__grouped_bar_uses_per_segment_colors() {
	svg := chart.new(width: 500, height: 300)
		.grouped_bar([[1.0, 2.0]], colors: ['#111111', '#222222'])
		.render()
	assert svg.contains('#111111')
	assert svg.contains('#222222')
}

fn test__grouped_bar_y_bound_is_max_segment_not_stack_sum() {
	// stacked_bar with [10, 10] would need a y-axis reaching 20; grouped_bar only
	// needs to reach 10 since bars sit side by side, not stacked.
	svg := chart.new(width: 500, height: 300)
		.grouped_bar([[10.0, 10.0]])
		.render()
	assert !svg.contains('>20<')
}
