module chart

import strings

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

pub fn render_svg(scene Scene, width int, height int, theme Theme) string {
	mut b := strings.new_builder(1024)
	b.write_string('<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">')
	b.write_string('<rect x="0" y="0" width="${width}" height="${height}" fill="${theme.background}"/>')
	for p in scene.primitives {
		b.write_string(primitive_to_svg(p))
	}
	b.write_string('</svg>')
	return b.str()
}
