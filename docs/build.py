#!/usr/bin/env python3
"""
docs/build.py — Generate docs/**/*.html from docs/src/**/*.md.
Run from repo root: python docs/build.py
"""
import os, re, glob
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.toc import TocExtension

DOCS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DOCS_DIR)
SRC_DIR   = os.path.join(DOCS_DIR, 'src')
OUT_DIR   = DOCS_DIR

SIDEBAR_ROOT = """\
<ul class="nav-list">
  <li><a href="index.html">Home</a></li>
  <li><a href="getting-started.html">Getting Started</a></li>
  <li><a href="concepts.html">Concepts</a></li>
  <li><a href="examples.html">Examples</a></li>
</ul>
<span class="nav-section-label">Module Reference</span>
<ul class="nav-list">
  <li><a href="modules/stats.html">stats</a></li>
  <li><a href="modules/experiment.html">experiment</a></li>
  <li><a href="modules/ml.html">ml</a></li>
  <li><a href="modules/hypothesis.html">hypothesis</a></li>
  <li><a href="modules/growth.html">growth</a></li>
  <li><a href="modules/nn.html">nn</a></li>
  <li><a href="modules/prob.html">prob</a></li>
  <li><a href="modules/optim.html">optim</a></li>
  <li><a href="modules/linalg.html">linalg</a></li>
  <li><a href="modules/utils.html">utils</a></li>
</ul>
<span class="nav-section-label">Resources</span>
<ul class="nav-list">
  <li><a href="companion/index.html">Companion Docs</a></li>
</ul>"""

SIDEBAR_SUB = SIDEBAR_ROOT \
    .replace('href="index.html"', 'href="../index.html"') \
    .replace('href="getting-started.html"', 'href="../getting-started.html"') \
    .replace('href="concepts.html"', 'href="../concepts.html"') \
    .replace('href="examples.html"', 'href="../examples.html"') \
    .replace('href="modules/', 'href="../modules/') \
    .replace('href="companion/', 'href="../companion/')


def make_page(title, body, depth=0):
    prefix  = '../' * depth
    nav     = SIDEBAR_ROOT if depth == 0 else SIDEBAR_SUB
    logo_href = f'{prefix}index.html'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — VStats</title>
<link rel="stylesheet" href="{prefix}css/style.css">
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-header">
      <a href="{logo_href}" class="logo"><span>V</span>Stats</a>
    </div>
    <nav class="sidebar-nav">
      {nav}
    </nav>
  </aside>
  <main class="content">
    {body}
  </main>
</div>
<script src="{prefix}js/script.js"></script>
</body>
</html>"""


def expand_includes(text):
    def repl(m):
        rel  = m.group(1).strip()
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            return f'<!-- include not found: {rel} -->'
        content = open(path).read().rstrip()
        return f'```v\n{content}\n```'
    return re.sub(r'<!--\s*include:\s*([^\s>]+)\s*-->', repl, text)


def to_html(md_text):
    md = markdown.Markdown(extensions=[
        FencedCodeExtension(), TableExtension(), TocExtension(permalink=False),
    ])
    return md.convert(md_text)


def src_to_out(src_path):
    rel  = os.path.relpath(src_path, SRC_DIR)
    base = os.path.splitext(rel)[0] + '.html'
    return os.path.join(OUT_DIR, base)


def page_depth(out_path):
    rel = os.path.relpath(out_path, OUT_DIR)
    return len(rel.split(os.sep)) - 1


def build_all():
    srcs = sorted(glob.glob(os.path.join(SRC_DIR, '**', '*.md'), recursive=True))
    print(f'Building {len(srcs)} pages from docs/src/ ...')
    for src in srcs:
        text  = open(src).read()
        text  = expand_includes(text)
        body  = to_html(text)
        m     = re.search(r'<h1[^>]*>(.*?)</h1>', body)
        title = re.sub(r'<[^>]+>', '', m.group(1)) if m else 'VStats'
        out   = src_to_out(src)
        depth = page_depth(out)
        html  = make_page(title, body, depth=depth)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, 'w').write(html)
        print(f'  ✓ {os.path.relpath(src, REPO_ROOT)} → {os.path.relpath(out, REPO_ROOT)}')
    print(f'Done. {len(srcs)} pages built.')


if __name__ == '__main__':
    build_all()
