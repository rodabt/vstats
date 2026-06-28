// Apply theme before paint to avoid flash
(function() {
    const saved = localStorage.getItem('vstats-theme');
    if (saved !== 'dark') {
        document.documentElement.setAttribute('data-theme', 'light');
    }
})();

document.addEventListener('DOMContentLoaded', function() {
    initSearch();
    highlightCurrentNav();
    initThemeToggle();
    initSyntaxHighlight();
});

function initSearch() {
    const searchInput = document.querySelector('.search-box input');
    if (!searchInput) return;

    searchInput.addEventListener('input', function(e) {
        const query = e.target.value.toLowerCase();
        if (!query) {
            clearHighlights();
            return;
        }
        searchDocument(query);
    });
}

function searchDocument(query) {
    const content = document.querySelector('.content');
    if (!content) return;

    clearHighlights();

    const walker = document.createTreeWalker(
        content,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    const nodesToHighlight = [];
    let node;
    while (node = walker.nextNode()) {
        if (node.textContent.toLowerCase().includes(query)) {
            nodesToHighlight.push(node);
        }
    }

    nodesToHighlight.forEach(node => {
        const parent = node.parentElement;
        if (parent && !parent.classList.contains('search-box')) {
            parent.classList.add('search-highlight');
        }
    });
}

function clearHighlights() {
    document.querySelectorAll('.search-highlight').forEach(el => {
        el.classList.remove('search-highlight');
    });
}

function highlightCurrentNav() {
    const currentPage = window.location.pathname.split('/').pop();
    const navLinks = document.querySelectorAll('.nav-list a');

    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPage) {
            link.classList.add('active');
        }
    });
}

function initThemeToggle() {
    const header = document.querySelector('.sidebar-header');
    if (!header) return;

    const btn = document.createElement('button');
    btn.className = 'theme-toggle';
    btn.setAttribute('aria-label', 'Toggle light/dark theme');
    btn.innerHTML = `
        <span class="theme-toggle-track"><span class="theme-toggle-thumb"></span></span>
        <span class="theme-toggle-label">Light</span>
    `;

    const isLight = () => document.documentElement.getAttribute('data-theme') === 'light';

    btn.addEventListener('click', function() {
        if (isLight()) {
            document.documentElement.removeAttribute('data-theme');
            localStorage.setItem('vstats-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            localStorage.setItem('vstats-theme', 'light');
        }
    });

    header.appendChild(btn);
}

// ── V Syntax Highlighting ─────────────────────────────────────────────────────

const V_KEYWORDS = new Set([
    'fn', 'pub', 'mut', 'if', 'else', 'for', 'in', 'return', 'import',
    'module', 'struct', 'interface', 'enum', 'match', 'or', 'true', 'false',
    'none', 'go', 'defer', 'unsafe', 'type', 'const', 'assert', 'break',
    'continue', 'as', 'is', 'lock', 'rlock', 'select', 'spawn', '__global',
    'static', 'volatile', 'union', 'map', 'chan', 'shared', 'atomic',
]);

const V_TYPES = new Set([
    'int', 'i8', 'i16', 'i32', 'i64', 'i128',
    'u8', 'u16', 'u32', 'u64', 'u128',
    'f32', 'f64', 'bool', 'string', 'rune', 'byte',
    'voidptr', 'byteptr', 'charptr', 'any', 'usize', 'isize',
]);

function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightV(src) {
    const out = [];
    let i = 0;
    const n = src.length;

    while (i < n) {
        const ch = src[i];

        // Block comment
        if (ch === '/' && src[i + 1] === '*') {
            const end = src.indexOf('*/', i + 2);
            const j = end === -1 ? n : end + 2;
            out.push('<span class="vs-comment">' + escHtml(src.slice(i, j)) + '</span>');
            i = j;
            continue;
        }

        // Line comment
        if (ch === '/' && src[i + 1] === '/') {
            const j = src.indexOf('\n', i);
            const end = j === -1 ? n : j;
            out.push('<span class="vs-comment">' + escHtml(src.slice(i, end)) + '</span>');
            i = end;
            continue;
        }

        // String (single-quote, double-quote, or backtick; skip escapes)
        if (ch === "'" || ch === '"' || ch === '`') {
            let j = i + 1;
            while (j < n) {
                if (src[j] === '\\' && ch !== '`') { j += 2; continue; }
                if (src[j] === ch) { j++; break; }
                j++;
            }
            out.push('<span class="vs-string">' + escHtml(src.slice(i, j)) + '</span>');
            i = j;
            continue;
        }

        // Compile-time variables and attributes: @WORD or @[...]
        if (ch === '@') {
            if (src[i + 1] === '[') {
                let j = i + 2;
                while (j < n && src[j] !== ']') j++;
                if (j < n) j++;
                out.push('<span class="vs-attr">' + escHtml(src.slice(i, j)) + '</span>');
                i = j;
            } else {
                let j = i + 1;
                while (j < n && /\w/.test(src[j])) j++;
                out.push('<span class="vs-attr">' + escHtml(src.slice(i, j)) + '</span>');
                i = j;
            }
            continue;
        }

        // Number: 0x hex, 0b binary, decimal integer or float
        if (/[0-9]/.test(ch)) {
            let j = i;
            if (src[j] === '0' && /[xX]/.test(src[j + 1])) {
                j += 2;
                while (j < n && /[0-9a-fA-F_]/.test(src[j])) j++;
            } else if (src[j] === '0' && /[bB]/.test(src[j + 1])) {
                j += 2;
                while (j < n && /[01_]/.test(src[j])) j++;
            } else {
                while (j < n && /[0-9_]/.test(src[j])) j++;
                if (j < n && src[j] === '.') {
                    j++;
                    while (j < n && /[0-9_]/.test(src[j])) j++;
                }
                if (j < n && /[eE]/.test(src[j])) {
                    j++;
                    if (j < n && /[+-]/.test(src[j])) j++;
                    while (j < n && /[0-9]/.test(src[j])) j++;
                }
            }
            out.push('<span class="vs-number">' + escHtml(src.slice(i, j)) + '</span>');
            i = j;
            continue;
        }

        // Word: keyword, built-in type, or plain identifier
        if (/[a-zA-Z_]/.test(ch)) {
            let j = i;
            while (j < n && /\w/.test(src[j])) j++;
            const word = src.slice(i, j);
            if (V_KEYWORDS.has(word)) {
                out.push('<span class="vs-keyword">' + escHtml(word) + '</span>');
            } else if (V_TYPES.has(word)) {
                out.push('<span class="vs-type">' + escHtml(word) + '</span>');
            } else {
                out.push(escHtml(word));
            }
            i = j;
            continue;
        }

        // Everything else: pass through escaped
        out.push(escHtml(ch));
        i++;
    }

    return out.join('');
}

function initSyntaxHighlight() {
    document.querySelectorAll('pre > code.language-v').forEach(function(block) {
        block.innerHTML = highlightV(block.textContent);
    });
}

// ─────────────────────────────────────────────────────────────────────────────

if (typeof searchIndex !== 'undefined') {
    function searchAPI(query) {
        return searchIndex.filter(item =>
            item.name.toLowerCase().includes(query) ||
            item.desc.toLowerCase().includes(query) ||
            item.module.toLowerCase().includes(query)
        );
    }
}
