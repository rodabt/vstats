module chart

@[params]
pub struct Theme {
pub:
	margin_left   int      = 60
	margin_right  int      = 20
	margin_top    int      = 40
	margin_bottom int      = 50
	background    string   = 'white'
	axis_color    string   = '#333333'
	axis_width    f64      = 1.0
	grid          bool // default false (Tufte: minimal ink)
	grid_color    string   = '#e0e0e0'
	font_family   string   = 'sans-serif'
	font_size     f64      = 12.0
	title_size    f64      = 16.0
	series_width  f64      = 1.5
	marker_radius  f64      = 3.0
	fill_opacity   f64      = 0.2
	subtitle_size  f64      = 12.0
	subtitle_color string   = '#666666'
	palette        []string = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}

pub fn (t Theme) color(i int) string {
	return t.palette[i % t.palette.len]
}
