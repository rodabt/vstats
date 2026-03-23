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

if (typeof searchIndex !== 'undefined') {
    function searchAPI(query) {
        return searchIndex.filter(item =>
            item.name.toLowerCase().includes(query) ||
            item.desc.toLowerCase().includes(query) ||
            item.module.toLowerCase().includes(query)
        );
    }
}
