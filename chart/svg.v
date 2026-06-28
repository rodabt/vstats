module chart

import strings
import math

fn xml_escape(s string) string {
	return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
}

fn meta_attrs(m Meta) string {
	if m.tooltip == '' {
		return ''
	}
	mut b := ''
	if m.series != '' {
		b += ' data-series="${xml_escape(m.series)}"'
	}
	if m.label != '' {
		b += ' data-label="${xml_escape(m.label)}"'
	}
	if m.x != '' {
		b += ' data-x="${xml_escape(m.x)}"'
	}
	if m.y != '' {
		b += ' data-y="${xml_escape(m.y)}"'
	}
	b += ' data-tooltip="${xml_escape(m.tooltip).replace('\n', '&#10;')}"'
	return b
}

fn meta_title(m Meta) string {
	if m.tooltip == '' {
		return ''
	}
	return '<title>${xml_escape(m.tooltip)}</title>'
}

fn primitive_to_svg(p Primitive) string {
	return match p {
		Line {
			'<line x1="${p.x1}" y1="${p.y1}" x2="${p.x2}" y2="${p.y2}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
		}
		Polyline {
			mut pts := []string{}
			for pt in p.points {
				pts << '${pt.x},${pt.y}'
			}
			'<polyline fill="none" stroke="${p.stroke}" stroke-width="${p.width}" points="${pts.join(' ')}"/>'
		}
		Rect {
			if p.meta.tooltip == '' {
				'<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
			} else {
				'<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${p.width}"${meta_attrs(p.meta)}>${meta_title(p.meta)}</rect>'
			}
		}
		Circle {
			if p.meta.tooltip == '' {
				'<circle cx="${p.cx}" cy="${p.cy}" r="${p.r}" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
			} else {
				'<circle cx="${p.cx}" cy="${p.cy}" r="${p.r}" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${p.width}"${meta_attrs(p.meta)}>${meta_title(p.meta)}</circle>'
			}
		}
		Text {
			anchor := match p.anchor {
				.start { 'start' }
				.middle { 'middle' }
				.end { 'end' }
			}
			mut transform := ''
			if p.rotate != 0.0 {
				transform = ' transform="rotate(${p.rotate} ${p.x} ${p.y})"'
			}
			'<text x="${p.x}" y="${p.y}" font-family="${p.family}" font-size="${p.size}" fill="${p.fill}" text-anchor="${anchor}"${transform}>${xml_escape(p.content)}</text>'
		}
		Polygon {
			mut pts := []string{}
			for pt in p.points {
				pts << '${pt.x},${pt.y}'
			}
			op := if p.opacity <= 0.0 { 1.0 } else { p.opacity }
			'<polygon points="${pts.join(' ')}" fill="${p.fill}" fill-opacity="${op}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
		}
	}
}

fn primitive_bounds(p Primitive) (f64, f64, f64, f64) {
	match p {
		Line {
			return math.min(p.x1, p.x2), math.min(p.y1, p.y2), math.max(p.x1, p.x2), math.max(p.y1,
				p.y2)
		}
		Rect {
			return p.x, p.y, p.x + p.w, p.y + p.h
		}
		Circle {
			return p.cx - p.r, p.cy - p.r, p.cx + p.r, p.cy + p.r
		}
		Polyline {
			mut minx := math.inf(1)
			mut miny := math.inf(1)
			mut maxx := math.inf(-1)
			mut maxy := math.inf(-1)
			for pt in p.points {
				if pt.x < minx {
					minx = pt.x
				}
				if pt.y < miny {
					miny = pt.y
				}
				if pt.x > maxx {
					maxx = pt.x
				}
				if pt.y > maxy {
					maxy = pt.y
				}
			}
			return minx, miny, maxx, maxy
		}
		Polygon {
			mut minx := math.inf(1)
			mut miny := math.inf(1)
			mut maxx := math.inf(-1)
			mut maxy := math.inf(-1)
			for pt in p.points {
				if pt.x < minx {
					minx = pt.x
				}
				if pt.y < miny {
					miny = pt.y
				}
				if pt.x > maxx {
					maxx = pt.x
				}
				if pt.y > maxy {
					maxy = pt.y
				}
			}
			return minx, miny, maxx, maxy
		}
		Text {
			w, h := text_extent(p.content, p.size)
			if p.rotate != 0.0 {
				// rotated ±90°: width/height swap around the anchor point
				return p.x - h, p.y - w, p.x + h, p.y + w
			}
			mut x0 := p.x
			mut x1 := p.x + w
			match p.anchor {
				.start {
					x0 = p.x
					x1 = p.x + w
				}
				.middle {
					x0 = p.x - w / 2.0
					x1 = p.x + w / 2.0
				}
				.end {
					x0 = p.x - w
					x1 = p.x
				}
			}
			return x0, p.y - h, x1, p.y + h * 0.3
		}
	}
}

pub fn render_svg(scene Scene, width int, height int, theme Theme) string {
	mut minx := 0.0
	mut miny := 0.0
	mut maxx := f64(width)
	mut maxy := f64(height)
	for p in scene.primitives {
		ax, ay, bx, by := primitive_bounds(p)
		if ax < minx {
			minx = ax
		}
		if ay < miny {
			miny = ay
		}
		if bx > maxx {
			maxx = bx
		}
		if by > maxy {
			maxy = by
		}
	}
	pad := 2.0
	if minx < 0.0 {
		minx -= pad
	}
	if miny < 0.0 {
		miny -= pad
	}
	if maxx > f64(width) {
		maxx += pad
	}
	if maxy > f64(height) {
		maxy += pad
	}
	vb_x := fmt_tick(minx)
	vb_y := fmt_tick(miny)
	vb_w := fmt_tick(maxx - minx)
	vb_h := fmt_tick(maxy - miny)

	mut b := strings.new_builder(1024)
	b.write_string('<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${vb_x} ${vb_y} ${vb_w} ${vb_h}">')
	b.write_string('<rect x="${vb_x}" y="${vb_y}" width="${vb_w}" height="${vb_h}" fill="${theme.background}"/>')
	for p in scene.primitives {
		b.write_string(primitive_to_svg(p))
	}
	b.write_string('</svg>')
	return b.str()
}
