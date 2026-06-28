module main

import chart
import os

fn main() {
	svg := chart.new(title: 'Quarterly Revenue', subtitle: 'hover the bars and points', width: 640,
		height: 420)
		.bar([120.0, 150.0, 170.0, 210.0], label: 'Revenue', labels: ['Q1', 'Q2', 'Q3', 'Q4'])
		.line([0.0, 1.0, 2.0, 3.0], [100.0, 160.0, 150.0, 230.0], label: 'Target')
		.render()

	// chart-interactions.js lives two directories up, in chart/
	html := '<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vstats chart tooltips demo</title>
<style>body{font-family:sans-serif;margin:2rem}</style>
</head>
<body>
<h1>vstats — interactive tooltips</h1>
<p>Move the cursor over the bars and the line vertices.</p>
${svg}
<script src="../../chart/chart-interactions.js"></script>
</body>
</html>
'
	out_dir := os.dir(os.executable())
	path := os.join_path(out_dir, 'index.html')
	os.write_file(path, html)!
	println('Wrote ${path}')
	println('Open it in a browser and hover the marks.')
}
