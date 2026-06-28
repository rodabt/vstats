// chart-interactions.js — vanilla, dependency-free rich tooltips for vstats charts.
// Include once in an HTML host that inlines chart SVGs:
//   <script src="chart-interactions.js"></script>
// It upgrades every [data-tooltip] mark into a styled, cursor-following tooltip.
(function (global) {
  'use strict';

  var el = null;

  function ensureEl() {
    if (el) return el;
    el = document.createElement('div');
    el.setAttribute('role', 'tooltip');
    var s = el.style;
    s.position = 'fixed';
    s.pointerEvents = 'none';
    s.zIndex = '9999';
    s.display = 'none';
    s.padding = '6px 8px';
    s.font = '12px sans-serif';
    s.lineHeight = '1.35';
    s.color = '#fff';
    s.background = 'rgba(20,20,20,0.92)';
    s.borderRadius = '4px';
    s.boxShadow = '0 1px 4px rgba(0,0,0,0.3)';
    s.maxWidth = '260px';
    s.whiteSpace = 'pre-line';
    document.body.appendChild(el);
    return el;
  }

  function show(text, x, y) {
    var t = ensureEl();
    t.textContent = text; // pre-line preserves newlines; textContent is XSS-safe
    t.style.display = 'block';
    move(x, y);
  }

  function move(x, y) {
    if (!el) return;
    var pad = 12;
    var w = el.offsetWidth;
    var h = el.offsetHeight;
    var left = x + pad;
    var top = y + pad;
    if (left + w > window.innerWidth) left = x - w - pad;
    if (top + h > window.innerHeight) top = y - h - pad;
    el.style.left = left + 'px';
    el.style.top = top + 'px';
  }

  function hide() {
    if (el) el.style.display = 'none';
  }

  function wire(node) {
    if (node.__cttWired) return;
    node.__cttWired = true;
    // strip native <title> so the browser default tooltip does not double-fire
    var title = node.querySelector('title');
    if (title) title.parentNode.removeChild(title);
    // getAttribute returns the already entity-decoded value, so &#10; is a real newline here
    var text = node.getAttribute('data-tooltip') || '';
    node.style.cursor = 'pointer';
    node.addEventListener('mouseenter', function (e) { show(text, e.clientX, e.clientY); });
    node.addEventListener('mousemove', function (e) { move(e.clientX, e.clientY); });
    node.addEventListener('mouseleave', hide);
  }

  function refresh(root) {
    var scope = root || document;
    var nodes = scope.querySelectorAll('[data-tooltip]');
    for (var i = 0; i < nodes.length; i++) wire(nodes[i]);
  }

  var ChartTooltips = { init: function () { refresh(document); }, refresh: refresh };
  global.ChartTooltips = ChartTooltips;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ChartTooltips.init);
  } else {
    ChartTooltips.init();
  }
})(window);
