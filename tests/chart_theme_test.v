import chart

fn test__theme_has_nonzero_defaults() {
	t := chart.Theme{}
	assert t.margin_left > 0
	assert t.font_size > 0.0
	assert t.series_width > 0.0
	assert t.palette.len > 0
	assert t.grid == false // Tufte: no grid by default
}

fn test__theme_color_cycles() {
	t := chart.Theme{}
	first := t.color(0)
	assert first.len > 0
	assert t.color(t.palette.len) == first // wraps around
}
