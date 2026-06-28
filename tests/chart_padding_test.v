import chart

fn test__long_y_label_grows_left_margin() {
	short := chart.new(width: 400, height: 300).bar([1.0, 2.0])
	long := chart.new(width: 400, height: 300).line([0.0, 1.0], [0.0, 1000000.0])
	sl, _, _, _ := short.effective_margins()
	ll, _, _, _ := long.effective_margins()
	assert ll > sl
}

fn test__adaptive_margins_preserve_requested_size() {
	svg := chart.new(width: 400, height: 300)
		.line([0.0, 1.0], [0.0, 1000000.0])
		.render()
	assert svg.contains('width="400"')
	assert svg.contains('height="300"')
}

fn test__effective_margins_never_below_theme_floor() {
	c := chart.new(width: 400, height: 300).bar([1.0])
	l, r, top, b := c.effective_margins()
	assert l >= 60
	assert r >= 20
	assert top >= 40
	assert b >= 50
}
