import chart

fn test__waterfall_emits_bars_and_labels() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([100.0, 20.0, -30.0, 90.0], ['Start', 'Gain', 'Loss', 'End'])
		.render()
	assert svg.count('<rect') >= 4 // background + 4 bars
	assert svg.contains('>Start<')
	assert svg.contains('>End<')
}

fn test__waterfall_uses_semantic_colors() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([100.0, 20.0, -30.0, 90.0], ['Start', 'Gain', 'Loss', 'End'])
		.render()
	assert svg.contains('#2ca02c') // increase (default)
	assert svg.contains('#d62728') // decrease (default)
	assert svg.contains('#7f7f7f') // total (default)
}

fn test__waterfall_color_overrides() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([10.0, 5.0, 15.0], ['Start', 'Gain', 'End'], color_increase: '#00ff00', color_total: '#000000')
		.render()
	assert svg.contains('#00ff00')
	assert svg.contains('#000000')
}

fn test__waterfall_show_values_labels_deltas() {
	svg := chart.new(width: 500, height: 300)
		.waterfall([100.0, 20.0, -30.0, 90.0], ['Start', 'Gain', 'Loss', 'End'], show_values: true)
		.render()
	assert svg.contains('>+20<')
	assert svg.contains('>-30<')
}

fn test__waterfall_two_point_totals_only() {
	svg := chart.new(width: 400, height: 300)
		.waterfall([50.0, 80.0], ['Start', 'End'])
		.render()
	assert svg.count('<rect') >= 2
}
