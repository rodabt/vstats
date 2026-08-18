import chart
import os

fn test__grid_tiles_two_charts_side_by_side() {
	c1 := chart.new(width: 300, height: 200).line([0.0, 1.0], [0.0, 1.0])
	c2 := chart.new(width: 300, height: 200).line([0.0, 1.0], [1.0, 0.0])
	g := chart.new_grid([c1, c2], cols: 2)
	svg := g.render()
	assert svg.starts_with('<svg')
	assert svg.count('<svg') == 3 // outer + 2 panels
	assert svg.contains('x="320"') // panel 2 offset = width(300) + gap(20)
}

fn test__grid_wraps_to_next_row() {
	c1 := chart.new(width: 200, height: 150).line([0.0, 1.0], [0.0, 1.0])
	c2 := chart.new(width: 200, height: 150).line([0.0, 1.0], [1.0, 0.0])
	c3 := chart.new(width: 200, height: 150).line([0.0, 1.0], [0.0, 1.0])
	g := chart.new_grid([c1, c2, c3], cols: 2)
	svg := g.render()
	assert svg.count('<svg') == 4 // outer + 3 panels
	assert svg.contains('y="170"') // row 2 offset = height(150) + gap(20)
}

fn test__grid_renders_figure_title() {
	c1 := chart.new(width: 200, height: 150).line([0.0, 1.0], [0.0, 1.0])
	g := chart.new_grid([c1], title: 'Overview')
	svg := g.render()
	assert svg.contains('Overview')
}

fn test__pair_plot_builds_nxn_grid_with_histogram_diagonal() {
	columns := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
	g := chart.pair_plot(columns, ['a', 'b'])
	assert g.charts.len == 4
	svg := g.render()
	assert svg.count('<svg') == 5 // outer + 4 panels
}

fn test__grid_save_writes_file() {
	c1 := chart.new(width: 100, height: 100).line([0.0, 1.0], [0.0, 1.0])
	g := chart.new_grid([c1])
	path := os.temp_dir() + '/vstats_grid_test.svg'
	g.save(path) or { assert false, 'save failed' }
	assert os.exists(path)
	os.rm(path) or {}
}
