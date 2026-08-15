import chart

fn test__secondary_axis_draws_right_side_ticks() {
	svg := chart.new(width: 500, height: 300)
		.bar([100.0, 200.0, 150.0])
		.line([0.0, 1.0, 2.0], [0.1, 0.5, 0.3], secondary_axis: true)
		.render()
	// right-anchored ticks only appear when a secondary axis is drawn
	assert svg.contains('text-anchor="start"')
	// secondary domain (0.1..0.5) tick should render, distinct from primary (100..200) domain
	assert svg.contains('>0.5<') || svg.contains('>0.50<')
}

fn test__no_secondary_axis_renders_unchanged() {
	svg := chart.new(width: 400, height: 300)
		.line([0.0, 1.0], [0.0, 1.0])
		.render()
	assert !svg.contains('text-anchor="start"') // no right-axis labels, no legend, no title
}

fn test__combo_chart_legend_renders_below_plot() {
	svg := chart.new(width: 500, height: 320)
		.bar([100.0, 200.0], label: 'Revenue')
		.line([0.0, 1.0], [0.1, 0.5], label: 'Conv. Rate', secondary_axis: true)
		.render()
	// legend swatch/text for a combo chart must sit below the plot area (y > plot bottom),
	// not in the right margin next to it
	legend_idx := svg.index('>Revenue<') or { panic('legend label not found') }
	prefix := svg[0..legend_idx]
	y_idx := prefix.last_index(' y="') or { panic('no y attr before legend label') }
	rest := prefix[y_idx + 4..]
	end := rest.index('"') or { panic('malformed y attr') }
	legend_y := rest[0..end].f64()
	assert legend_y > 200.0 // plot bottom sits well above this given height:320 + margins
}

fn test__secondary_axis_series_uses_independent_scale() {
	// primary series spans 0..1000, secondary spans 0..1 -- if they shared one scale,
	// the secondary line would be squashed to a flat near-zero line. Assert both
	// axes' extreme tick values render distinctly.
	svg := chart.new(width: 500, height: 300)
		.bar([0.0, 1000.0])
		.line([0.0, 1.0], [0.0, 1.0], secondary_axis: true)
		.render()
	assert svg.contains('>1000<')
	assert svg.contains('>1<')
}
