#!/usr/bin/env python3
"""Build docs/companion/index.html. Run from docs/companion/."""

import re, os
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.toc import TocExtension

CHAPTERS = [
    ("00-foundations.md",       "00", "Foundations"),
    ("01-design.md",            "01", "Experiment Design"),
    ("02-data-quality.md",      "02", "Data Quality"),
    ("03-metric-pitfalls.md",   "03", "Metric Pitfalls"),
    ("04-variance-reduction.md","04", "Variance Reduction"),
    ("05-causal-methods.md",    "05", "Causal Methods"),
    ("06-readout.md",           "06", "Readout"),
    ("07-communication.md",     "07", "Communication"),
]

SECTION_LABELS = {
    "when to use": ("WHEN", "lbl-when"),
    "why":         ("WHY",  "lbl-why"),
    "how":         ("HOW",  "lbl-how"),
    "pitfalls":    ("PITFALLS", "lbl-pitfalls"),
    "python":      ("PYTHON",   "lbl-python"),
    "sql":         ("SQL",      "lbl-sql"),
    "vstats (future)": ("VSTATS", "lbl-vstats"),
}

def slugify(t):
    t = t.lower()
    t = re.sub(r'[^\w\s-]', '', t)
    t = re.sub(r'[\s_]+', '-', t)
    return t.strip('-')

def extract_cards(path):
    cards = []
    with open(path) as f:
        for line in f:
            m = re.match(r'^## (.+)$', line.strip())
            if m and not m.group(1).startswith('['):
                title = m.group(1)
                cards.append((title, slugify(title)))
    return cards

def to_html(md_text):
    md = markdown.Markdown(extensions=[
        TableExtension(), FencedCodeExtension(),
        TocExtension(permalink=False),
    ])
    return md.convert(md_text)

def transform(html):
    # h2 → recipe card wrapper with anchor
    def h2(m):
        raw = m.group(1)
        raw = re.sub(r'<a[^>]*>.*?</a>', '', raw, flags=re.DOTALL).strip()
        text = re.sub(r'<[^>]+>', '', raw).strip()
        slug = slugify(text)
        return (
            '</div>\n'
            f'<div class="card" id="{slug}">\n'
            f'<h2><a href="#{slug}" class="anchor" aria-hidden="true">#</a>{text}</h2>'
        )
    html = re.sub(r'<h2[^>]*>(.*?)</h2>', h2, html, flags=re.DOTALL)
    html = html.lstrip()
    if html.startswith('</div>'):
        html = html[len('</div>\n'):]
    html += '\n</div>'

    # h3 → labeled section header
    def h3(m):
        raw = m.group(1)
        raw = re.sub(r'<a[^>]*>.*?</a>', '', raw, flags=re.DOTALL).strip()
        text = re.sub(r'<[^>]+>', '', raw).strip()
        key = text.lower()
        label, cls = SECTION_LABELS.get(key, (text.upper(), 'lbl-default'))
        return f'<h3 class="sec-label {cls}">{label}</h3>'
    html = re.sub(r'<h3[^>]*>(.*?)</h3>', h3, html, flags=re.DOTALL)

    # blockquote → tagline or vstats placeholder
    def bq(m):
        inner = m.group(1).strip()
        return f'<div class="tagline">{inner}</div>'
    html = re.sub(r'<blockquote>\s*(<p>.*?</p>)\s*</blockquote>', bq, html, flags=re.DOTALL)

    # vstats tagline → placeholder style
    html = re.sub(
        r'(lbl-vstats">[^<]*</h3>\s*)<div class="tagline">',
        r'\1<div class="vstats-note">',
        html, flags=re.DOTALL
    )

    # code blocks → wrap with lang label
    def code_block(m):
        cls = m.group(1) or ''
        body = m.group(2)
        lang = ''
        if 'python' in cls:  lang = 'Python'
        elif 'sql' in cls:   lang = 'SQL'
        label = f'<span class="code-lang">{lang}</span>' if lang else ''
        return f'<div class="code-wrap">{label}<pre><code class="{cls}">{body}</code></pre></div>'
    html = re.sub(r'<pre><code(?: class="([^"]*)")?>(.*?)</code></pre>', code_block, html, flags=re.DOTALL)

    # images → add class
    html = re.sub(r'<img ([^>]*src="charts/[^"]*"[^>]*)/>', r'<img class="chart" \1/>', html)

    return html

# ── Build ─────────────────────────────────────────────────────────────────────

all_chapters = []
for fname, num, title in CHAPTERS:
    cards = extract_cards(fname)
    with open(fname) as f:
        md = re.sub(r'^# .+\n', '', f.read(), count=1)
    html = transform(to_html(md))
    all_chapters.append((num, title, cards, html))

# sidebar
sidebar_items = []
for num, title, cards, _ in all_chapters:
    links = '\n'.join(
        f'<li><a href="#{slug}" class="card-link">{name}</a></li>'
        for name, slug in cards
    )
    sidebar_items.append(f'''
<details class="ch-group">
  <summary><a href="#ch-{num}">{num} — {title}</a></summary>
  <ul>{links}</ul>
</details>''')

# content
content_parts = []
for num, title, cards, body in all_chapters:
    content_parts.append(f'''
<section id="ch-{num}" class="chapter">
  <div class="ch-header">
    <span class="ch-num">{num}</span>
    <h1>{title}</h1>
  </div>
  {body}
</section>''')

total_cards = sum(len(c[2]) for c in all_chapters)

PAGE = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SaaS Experiment Companion</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --sidebar: 260px;
  --bg: #fff;
  --surface: #f8f8f7;
  --border: #e8e8e6;
  --text: #111;
  --muted: #888;
  --accent: #1d6fb8;
  --accent-bg: #f0f6ff;
  --code-bg: #f5f5f4;
  --when: #0e7490;
  --why: #166534;
  --how: #92400e;
  --pitfalls: #991b1b;
  --python: #1e40af;
  --sql: #5b21b6;
  --vstats: #6b7280;
}}

html {{ scroll-behavior: smooth; font-size: 15px; }}

body {{
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  display: flex;
  min-height: 100vh;
}}

/* ── Sidebar ── */
#sidebar {{
  width: var(--sidebar);
  position: fixed;
  top: 0; left: 0; bottom: 0;
  overflow-y: auto;
  border-right: 1px solid var(--border);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  font-size: 0.825rem;
}}

#sidebar-top {{
  padding: 1.25rem 1rem 0.75rem;
  border-bottom: 1px solid var(--border);
}}

#sidebar-top h2 {{
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.15rem;
  letter-spacing: 0;
  border: none;
  padding: 0;
}}

#sidebar-top p {{
  font-size: 0.72rem;
  color: var(--muted);
}}

#search {{
  width: 100%;
  margin-top: 0.6rem;
  padding: 0.35rem 0.6rem;
  border: 1px solid var(--border);
  background: var(--bg);
  font-family: inherit;
  font-size: 0.78rem;
  color: var(--text);
  outline: none;
}}

#search:focus {{ border-color: var(--accent); }}

#nav {{ flex: 1; padding: 0.5rem 0; overflow-y: auto; }}

.ch-group {{ margin: 0; }}

.ch-group summary {{
  list-style: none;
  padding: 0.45rem 1rem;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.78rem;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 0.4rem;
  user-select: none;
}}

.ch-group summary::-webkit-details-marker {{ display: none; }}
.ch-group summary::before {{ content: '›'; color: var(--muted); transition: transform 0.15s; display: inline-block; }}
.ch-group[open] summary::before {{ transform: rotate(90deg); }}
.ch-group summary:hover {{ color: var(--accent); }}
.ch-group summary a {{ color: inherit; text-decoration: none; }}

.ch-group ul {{ list-style: none; padding: 0 0 0.25rem 1.5rem; }}

.card-link {{
  display: block;
  padding: 0.22rem 0.5rem;
  color: var(--muted);
  text-decoration: none;
  border-left: 2px solid transparent;
  transition: color 0.1s;
  font-size: 0.77rem;
  line-height: 1.4;
}}

.card-link:hover, .card-link.active {{
  color: var(--accent);
  border-left-color: var(--accent);
}}

/* ── Content ── */
#content {{
  margin-left: var(--sidebar);
  max-width: calc(var(--sidebar) + 780px);
  padding: 3rem 4rem 6rem;
  width: 100%;
}}

/* ── Chapter ── */
.chapter {{ margin-bottom: 4rem; }}

.ch-header {{
  display: flex;
  align-items: baseline;
  gap: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid var(--border);
  margin-bottom: 2.5rem;
}}

.ch-num {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: var(--muted);
  font-weight: 500;
  flex-shrink: 0;
}}

.chapter h1 {{
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--text);
}}

/* ── Recipe card ── */
.card {{
  margin-bottom: 3rem;
  scroll-margin-top: 1.5rem;
}}

.card h2 {{
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
  padding: 0.5rem 0.75rem;
  background: var(--surface);
  border-left: 3px solid var(--accent);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}}

.anchor {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--border);
  text-decoration: none;
  flex-shrink: 0;
}}

.anchor:hover {{ color: var(--accent); }}

/* tagline */
.tagline {{
  border-left: 3px solid var(--accent);
  padding: 0.5rem 0.9rem;
  margin-bottom: 1.25rem;
  background: var(--accent-bg);
}}

.tagline p {{
  color: var(--text);
  font-size: 0.9rem;
  font-style: italic;
  margin: 0;
}}

/* section labels */
.sec-label {{
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  padding: 0.2rem 0.5rem;
  margin-top: 1.5rem;
  margin-bottom: 0.5rem;
  display: inline-block;
  border-radius: 2px;
}}

.lbl-when    {{ background: #ecfeff; color: var(--when);    border: 1px solid #a5f3fc; }}
.lbl-why     {{ background: #f0fdf4; color: var(--why);     border: 1px solid #bbf7d0; }}
.lbl-how     {{ background: #fffbeb; color: var(--how);     border: 1px solid #fde68a; }}
.lbl-pitfalls{{ background: #fef2f2; color: var(--pitfalls);border: 1px solid #fecaca; }}
.lbl-python  {{ background: #eff6ff; color: var(--python);  border: 1px solid #bfdbfe; }}
.lbl-sql     {{ background: #f5f3ff; color: var(--sql);     border: 1px solid #ddd6fe; }}
.lbl-vstats  {{ background: #f9fafb; color: var(--vstats);  border: 1px solid var(--border); }}
.lbl-default {{ background: var(--surface); color: var(--muted); border: 1px solid var(--border); }}

/* vstats placeholder */
.vstats-note {{
  border: 1px dashed var(--border);
  padding: 0.5rem 0.9rem;
  margin: 0.25rem 0 1.5rem;
}}

.vstats-note p {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--muted);
  margin: 0;
}}

/* prose */
.card p {{
  font-size: 0.9rem;
  color: #333;
  margin-bottom: 0.75rem;
}}

.card ul, .card ol {{
  padding-left: 1.4rem;
  margin-bottom: 0.75rem;
}}

.card li {{
  font-size: 0.9rem;
  color: #333;
  margin-bottom: 0.2rem;
  line-height: 1.6;
}}

.card li strong {{ color: var(--text); }}

.card a {{ color: var(--accent); }}

/* tables */
.card table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  margin: 0.75rem 0 1.25rem;
}}

.card th {{
  background: var(--surface);
  text-align: left;
  padding: 0.45rem 0.75rem;
  font-weight: 600;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
}}

.card td {{
  padding: 0.4rem 0.75rem;
  border-bottom: 1px solid var(--border);
  color: #444;
  vertical-align: top;
}}

.card tr:last-child td {{ border-bottom: none; }}
.card tr:hover td {{ background: var(--surface); }}

/* code */
.code-wrap {{
  margin: 0.5rem 0 1.25rem;
}}

.code-lang {{
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.63rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--surface);
  border: 1px solid var(--border);
  border-bottom: none;
  padding: 0.15rem 0.5rem;
}}

.code-wrap pre {{
  background: var(--code-bg) !important;
  border: 1px solid var(--border);
  border-radius: 0;
  padding: 1rem !important;
  overflow-x: auto;
  margin: 0 !important;
  box-shadow: none !important;
}}

.code-wrap code {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  line-height: 1.6 !important;
  background: none !important;
  padding: 0 !important;
  border: none !important;
}}

/* prism overrides — keep it clean on white */
code[class*="language-"], pre[class*="language-"] {{
  background: var(--code-bg) !important;
  text-shadow: none !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
}}

/* inline code */
.card p code, .card li code, .card td code {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8em;
  background: var(--code-bg);
  border: 1px solid var(--border);
  padding: 0.1em 0.35em;
  color: #c2410c;
}}

/* charts */
.chart {{
  display: block;
  width: 100%;
  max-width: 100%;
  margin: 0.75rem 0 1.5rem;
  border: 1px solid var(--border);
}}

/* hr */
.card hr {{ border: none; border-top: 1px solid var(--border); margin: 2rem 0; }}

/* ── Responsive ── */
@media (max-width: 820px) {{
  #sidebar {{ position: relative; width: 100%; height: auto; border-right: none; border-bottom: 1px solid var(--border); }}
  #content {{ margin-left: 0; padding: 2rem 1.5rem 4rem; }}
  body {{ flex-direction: column; }}
}}
</style>
</head>
<body>

<aside id="sidebar">
  <div id="sidebar-top">
    <h2>Experiment Companion</h2>
    <p>{total_cards} cards · Python · SQL</p>
    <input id="search" type="search" placeholder="search cards…" autocomplete="off">
  </div>
  <nav id="nav">
{''.join(sidebar_items)}
  </nav>
</aside>

<main id="content">
{''.join(content_parts)}
</main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
<script>
// search
document.getElementById('search').addEventListener('input', function() {{
  const q = this.value.toLowerCase();
  document.querySelectorAll('.card-link').forEach(a => {{
    a.style.display = (!q || a.textContent.toLowerCase().includes(q)) ? '' : 'none';
  }});
  if (q) document.querySelectorAll('.ch-group').forEach(g => g.open = true);
}});

// scroll spy
(function() {{
  const cards = [...document.querySelectorAll('.card[id]')];
  const links = document.querySelectorAll('.card-link');
  function update() {{
    let cur = null;
    for (const c of cards) {{
      if (c.getBoundingClientRect().top <= 100) cur = c.id;
    }}
    links.forEach(a => {{
      const on = a.getAttribute('href') === '#' + cur;
      a.classList.toggle('active', on);
      if (on) a.closest('details').open = true;
    }});
  }}
  window.addEventListener('scroll', update, {{passive: true}});
  update();
}})();
</script>
</body>
</html>
'''

with open('index.html', 'w') as f:
    f.write(PAGE)

size = os.path.getsize('index.html')
print(f"✓ index.html  ({size//1024} KB, {total_cards} cards)")
