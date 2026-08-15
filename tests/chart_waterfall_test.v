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

fn find_text_y(svg string, content string) f64 {
	marker := '>${content}<'
	idx := svg.index(marker) or { panic('marker not found: ${marker}') }
	prefix := svg[0..idx]
	y_idx := prefix.last_index(' y="') or { panic('no y attr before marker: ${marker}') }
	rest := prefix[y_idx + 4..]
	end := rest.index('"') or { panic('malformed y attr') }
	return rest[0..end].f64()
}

fn find_rect_y_by_tooltip(svg string, snippet string) f64 {
	idx := svg.index(snippet) or { panic('tooltip snippet not found: ${snippet}') }
	prefix := svg[0..idx]
	rect_idx := prefix.last_index('<rect') or { panic('no rect before snippet: ${snippet}') }
	after := svg[rect_idx..]
	y_idx := after.index(' y="') or { panic('no y attr in rect for: ${snippet}') }
	rest := after[y_idx + 4..]
	end := rest.index('"') or { panic('malformed y attr') }
	return rest[0..end].f64()
}

fn test__waterfall_label_always_above_segment() {
	// 'Drop' is a negative delta: cumulative goes 137 -> 85. Its bar's own top (smallest
	// pixel y) corresponds to the HIGHER value (137, the pre-delta cumulative), so the
	// label must sit just above that top edge -- not near the lower post-delta edge.
	svg := chart.new(width: 400, height: 300)
		.waterfall([137.0, -52.0, 85.0], ['Start', 'Drop', 'End'], show_values: true)
		.render()
	drop_label_y := find_text_y(svg, '-52')
	drop_rect_y := find_rect_y_by_tooltip(svg, 'Drop: -52')
	assert drop_label_y < drop_rect_y
	assert drop_rect_y - drop_label_y < 10.0
}
