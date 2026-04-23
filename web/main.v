module main

import veb
import os

pub struct Context {
	veb.Context
}

pub struct App {
	veb.StaticHandler
}

fn main() {
	mut app := &App{}
	static_dir := os.join_path(@VMODROOT, 'web', 'static')
	app.mount_static_folder_at(static_dir, '/static') or { panic(err) }
	veb.run[App, Context](mut app, 8080)
}
