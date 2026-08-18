module chart

import strings
import os

@[params]
pub struct GridOpts {
pub:
	cols  int = 2
	gap   int = 20
	title string
}

pub struct Grid {
pub mut:
	charts []Chart
	opts   GridOpts
}

pub fn new_grid(charts []Chart, opts GridOpts) Grid {
	return Grid{
		charts: charts
		opts:   opts
	}
}

// pair_plot builds an N x N scatter-matrix Grid over `columns` (one []f64 per
// variable, matched by index to `labels`): off-diagonal panels are pairwise
// scatter plots, diagonal panels are that column's histogram.
pub fn pair_plot(columns [][]f64, labels []string, opts GridOpts) Grid {
	assert columns.len == labels.len
	n := columns.len
	mut charts := []Chart{cap: n * n}
	for row in 0 .. n {
		for col in 0 .. n {
			mut c := new(width: 320, height: 240, title: '${labels[col]} vs ${labels[row]}')
			if row == col {
				c = c.histogram(columns[col])
			} else {
				c = c.scatter(columns[col], columns[row])
			}
			charts << c
		}
	}
	return new_grid(charts, cols: n, gap: opts.gap, title: opts.title)
}

pub fn (g Grid) render() string {
	cols := if g.opts.cols > 0 { g.opts.cols } else { 1 }
	gap := g.opts.gap
	rows := (g.charts.len + cols - 1) / cols

	mut col_widths := []int{len: cols}
	mut row_heights := []int{len: rows}
	for i, c in g.charts {
		row := i / cols
		col := i % cols
		if c.width > col_widths[col] {
			col_widths[col] = c.width
		}
		if c.height > row_heights[row] {
			row_heights[row] = c.height
		}
	}

	mut col_x := []int{len: cols}
	mut acc_x := 0
	for col in 0 .. cols {
		col_x[col] = acc_x
		acc_x += col_widths[col] + gap
	}
	mut row_y := []int{len: rows}
	mut acc_y := 0
	title_offset := if g.opts.title != '' { 32 } else { 0 }
	for row in 0 .. rows {
		row_y[row] = acc_y
		acc_y += row_heights[row] + gap
	}

	total_w := if acc_x > 0 { acc_x - gap } else { 0 }
	total_h := (if acc_y > 0 { acc_y - gap } else { 0 }) + title_offset

	mut b := strings.new_builder(1024)
	b.write_string('<svg xmlns="http://www.w3.org/2000/svg" width="${total_w}" height="${total_h}">')
	if g.opts.title != '' {
		b.write_string('<text x="${total_w / 2}" y="20" font-size="16" text-anchor="middle" font-family="sans-serif">${g.opts.title}</text>')
	}
	for i, c in g.charts {
		row := i / cols
		col := i % cols
		px := col_x[col]
		py := row_y[row] + title_offset
		panel_svg := c.render()
		positioned := panel_svg[..4] + ' x="${px}" y="${py}"' + panel_svg[4..]
		b.write_string(positioned)
	}
	b.write_string('</svg>')
	return b.str()
}

pub fn (g Grid) save(path string) ! {
	os.write_file(path, g.render())!
}
