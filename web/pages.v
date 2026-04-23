module main

import veb
import os

const tmpl_dir = @VMODROOT + '/web/templates'

fn read_template(name string) !string {
	header := os.read_file('${tmpl_dir}/_header.html')!
	footer := os.read_file('${tmpl_dir}/_footer.html')!
	body := os.read_file('${tmpl_dir}/${name}.html')!
	return header + body + footer
}

@['/'; get]
pub fn (app &App) index(mut ctx Context) veb.Result {
	html := read_template('index') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/ab-test'; get]
pub fn (app &App) ab_test_page(mut ctx Context) veb.Result {
	html := read_template('ab_test') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/sprt'; get]
pub fn (app &App) sprt_page(mut ctx Context) veb.Result {
	html := read_template('sprt') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/power-analysis'; get]
pub fn (app &App) power_analysis_page(mut ctx Context) veb.Result {
	html := read_template('power_analysis') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/cuped'; get]
pub fn (app &App) cuped_page(mut ctx Context) veb.Result {
	html := read_template('cuped') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/hypothesis'; get]
pub fn (app &App) hypothesis_page(mut ctx Context) veb.Result {
	html := read_template('hypothesis') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/psm'; get]
pub fn (app &App) psm_page(mut ctx Context) veb.Result {
	html := read_template('psm') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/did'; get]
pub fn (app &App) did_page(mut ctx Context) veb.Result {
	html := read_template('did') or { return ctx.server_error('template error') }
	return ctx.html(html)
}
