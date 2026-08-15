import chart

fn test__dumbbell_emits_two_circles_and_connector_per_row() {
	svg := chart.new(width: 500, height: 300)
		.dumbbell([10.0, 20.0, 15.0], [25.0, 22.0, 30.0], ['Q1', 'Q2', 'Q3'])
		.render()
	assert svg.count('<circle') == 6 // 3 rows * 2 dots
	assert svg.contains('>Q1<')
	assert svg.contains('>Q2<')
	assert svg.contains('>Q3<')
}

fn test__dumbbell_before_after_distinct_colors() {
	svg := chart.new(width: 500, height: 300)
		.dumbbell([10.0], [20.0], ['Only'], color: '#111111', color_hi: '#222222')
		.render()
	assert svg.contains('#111111')
	assert svg.contains('#222222')
}

fn test__dumbbell_show_values_labels_both_points() {
	svg := chart.new(width: 500, height: 300)
		.dumbbell([10.0], [25.0], ['Row'], show_values: true)
		.render()
	assert svg.contains('>10<')
	assert svg.contains('>25<')
}

fn test__xmin_forces_x_axis_start() {
	// without xmin(), the x-axis would start near the data min (180); xmin(100) should
	// force the domain to start at 100 instead, shifting where value 180 lands on-screen.
	without := chart.new(width: 500, height: 300)
		.dumbbell([320.0, 410.0], [180.0, 260.0], ['A', 'B'])
		.render()
	with_override := chart.new(width: 500, height: 300)
		.xmin(100.0)
		.dumbbell([320.0, 410.0], [180.0, 260.0], ['A', 'B'])
		.render()
	assert with_override.contains('>100<')
	assert without != with_override
}
