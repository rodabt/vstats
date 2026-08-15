import chart

fn test__xcategories_renders_labels_on_x_axis() {
	svg := chart.new(width: 400, height: 300)
		.xcategories(['Jan', 'Feb', 'Mar'])
		.line([0.0, 1.0, 2.0], [10.0, 20.0, 15.0])
		.render()
	assert svg.contains('>Jan<')
	assert svg.contains('>Feb<')
	assert svg.contains('>Mar<')
	// numeric ticks should not appear for the x-axis when categories are set
	assert !svg.contains('>0.5<')
}

fn test__xcategories_works_with_bar() {
	svg := chart.new(width: 400, height: 300)
		.xcategories(['A', 'B'])
		.bar([5.0, 9.0])
		.render()
	assert svg.contains('>A<')
	assert svg.contains('>B<')
}

fn test__no_xcategories_keeps_numeric_ticks() {
	svg := chart.new(width: 400, height: 300)
		.line([0.0, 1.0, 2.0], [0.0, 50.0, 100.0])
		.render()
	assert svg.contains('>100<') // numeric tick, unaffected by this feature
}

fn test__xcategories_does_not_break_existing_stacked_bar_labels() {
	// stacked_bar's own per-series labels fallback still works when xcategories_ unset
	svg := chart.new(width: 400, height: 300)
		.stacked_bar([[1.0, 2.0], [3.0, 4.0]], labels: ['G1', 'G2'])
		.render()
	assert svg.contains('>G1<')
	assert svg.contains('>G2<')
}
