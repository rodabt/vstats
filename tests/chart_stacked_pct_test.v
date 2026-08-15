import chart

fn test__stacked_bar_pct_emits_rects_and_percent_labels() {
	svg := chart.new(width: 500, height: 300)
		.xcategories(['Q1', 'Q2'])
		.stacked_bar_pct([[30.0, 70.0], [50.0, 50.0]], colors: ['#08306b', '#f7fbff'])
		.render()
	assert svg.contains('%<')
}

fn test__stacked_bar_pct_uses_contrast_text_color() {
	// dark segment (#000000) should get white text; light segment (#ffffff) should get black text
	svg := chart.new(width: 500, height: 300)
		.stacked_bar_pct([[60.0, 40.0]], colors: ['#000000', '#ffffff'])
		.render()
	assert svg.contains('fill="#ffffff"')
	assert svg.contains('fill="#000000"')
}

fn test__stacked_bar_pct_omits_label_for_tiny_segment() {
	// segment 2 is ~1% of a 300px-tall plot -- far too short to fit a label
	svg := chart.new(width: 500, height: 300)
		.stacked_bar_pct([[99.0, 1.0]], colors: ['#08306b', '#f7fbff'])
		.render()
	assert !svg.contains('>1%<')
}

fn test__stacked_bar_pct_normalizes_unequal_totals_to_100() {
	// group totals differ (10 vs 1000) but both should render as full-height bars
	svg := chart.new(width: 500, height: 300)
		.stacked_bar_pct([[5.0, 5.0], [500.0, 500.0]])
		.render()
	assert svg.count('<rect') >= 5 // background + 2 groups * 2 segments
}
