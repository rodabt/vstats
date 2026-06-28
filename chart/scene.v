module chart

pub struct Point {
pub:
	x f64
	y f64
}

pub struct Meta {
pub:
	tooltip string
	series  string
	label   string
	x       string
	y       string
}

pub enum TextAnchor {
	start
	middle
	end
}

pub struct Line {
pub:
	x1     f64
	y1     f64
	x2     f64
	y2     f64
	stroke string
	width  f64
}

pub struct Polyline {
pub:
	points []Point
	stroke string
	width  f64
}

pub struct Rect {
pub:
	x      f64
	y      f64
	w      f64
	h      f64
	fill   string
	stroke string
	width  f64
	meta   Meta
}

pub struct Circle {
pub:
	cx     f64
	cy     f64
	r      f64
	fill   string
	stroke string
	width  f64
	meta   Meta
}

pub struct Text {
pub:
	x       f64
	y       f64
	content string
	size    f64
	fill    string
	anchor  TextAnchor
	family  string
	rotate  f64 // degrees; 0 = horizontal
}

pub struct Polygon {
pub:
	points  []Point
	fill    string
	opacity f64
	stroke  string
	width   f64
}

pub type Primitive = Line | Polyline | Rect | Circle | Text | Polygon

pub struct Scene {
pub mut:
	primitives []Primitive
}
