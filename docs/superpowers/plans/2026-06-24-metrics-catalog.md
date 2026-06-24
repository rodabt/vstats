# Metrics Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/metrics` catalog page to the tracker where teams manage reusable metrics (name, slug, type, DuckDB SQL) and refresh their baseline values on demand.

**Architecture:** SQLite stores metric definitions and cached baseline values; a vduckdb in-memory connection executes the metric's multi-statement SQL on refresh and writes the scalar result back to SQLite. The Alpine SPA adds a `metricsPage` component and store wiring mirroring the existing `/settings` pattern.

**Tech Stack:** V + veb (backend), SQLite (storage), vduckdb (baseline execution), Alpine.js 3 (frontend), vanilla CSS

## Global Constraints

- All backend code lives in `apps/tracker/main.v` — do not create new V files
- All frontend changes are in `apps/tracker/public/app.js`, `apps/tracker/public/index.html`, `apps/tracker/public/app.css`
- `apps/` is git-ignored — no commits; verify by curl/browser instead
- vduckdb pattern: `mut vdb := vduckdb.DuckDB{}` → `vdb.open(':memory:')!` → `vdb.query(stmt)!` per statement → `vdb.get_first_row()` → `vdb.close()` (via `defer`)
- `LIBDUCKDB_DIR` env var points to `/home/rabt/devel/vduckdb/src/thirdparty` — already set; do not change
- Duplicate detection uses pre-insert SELECT (not relying on UNIQUE constraint), same pattern as `flagging_systems`
- `metric_type` must be exactly `'proportion'` or `'continuous'`
- Slug is immutable after creation — PUT endpoint ignores any slug field in the request body
- Server runs on port 8080 (`v run .` from `apps/tracker/`)

---

### Task 1: Backend — metrics table, 5 API endpoints, vduckdb refresh

**Files:**
- Modify: `apps/tracker/v.mod` — add vduckdb dependency
- Modify: `apps/tracker/main.v` — import vduckdb, migration, structs, `run_metric_sql`, 5 handlers, `/metrics` route

**Interfaces:**
- Produces:
  - `GET /api/metrics` → `[{"id","name","slug","metric_type","sql_query","baseline_value","baseline_updated_at"}]`
  - `POST /api/metrics` body `{"name","slug","metric_type","sql_query"}` → 201 item or 400/409
  - `PUT /api/metrics/:id` body `{"name","metric_type","sql_query"}` → 200 item or 400
  - `DELETE /api/metrics/:id` → 200 "deleted"
  - `POST /api/metrics/:id/refresh` → 200 `{"value","updated_at"}` or 422 `{"error":"..."}`
  - `GET /metrics` → 200 (serves index.html)

- [ ] **Step 1: Add vduckdb to v.mod**

Open `apps/tracker/v.mod` and replace its contents with:

```
Module {
	name: 'tracker'
	 description: 'Experiment Tracker web app
	 version: '0.0.1'
	 dependencies: [
		{ name: 'veb' }
		{ name: 'db.sqlite' }
		{ name: 'vstats' }
		{ name: 'vduckdb' }
	]
}
```

- [ ] **Step 2: Add import and structs to main.v**

At the top of `apps/tracker/main.v`, add `vduckdb` to the import block:

```v
import veb
import db.sqlite
import json
import time
import vstats.experiment
import vduckdb
```

After the existing `SettingsItemReq` struct (search for it near the bottom of the structs section), add:

```v
struct MetricCreateReq {
	name        string
	slug        string
	metric_type string
	sql_query   string
}

struct MetricUpdateReq {
	name        string
	metric_type string
	sql_query   string
}
```

- [ ] **Step 3: Add metrics migration to run_migrations()**

Inside `run_migrations()`, append one more entry to the `migrations` array (after the `tracking_tools` line):

```v
'CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, slug TEXT NOT NULL UNIQUE, metric_type TEXT NOT NULL DEFAULT \'proportion\', sql_query TEXT NOT NULL DEFAULT \'\', baseline_value REAL, baseline_updated_at TEXT)',
```

- [ ] **Step 4: Add run_metric_sql helper**

Add this function anywhere before the handler functions (e.g., after `get_current_time`):

```v
fn run_metric_sql(sql_query string) !(string, string) {
	mut vdb := vduckdb.DuckDB{}
	defer {
		vdb.close()
	}
	_ := vdb.open(':memory:')!
	statements := sql_query.split(';').map(it.trim_space()).filter(it != '')
	if statements.len == 0 {
		return error('empty query')
	}
	for stmt in statements {
		_ := vdb.query(stmt)!
	}
	row := vdb.get_first_row()
	if row.len == 0 {
		return error('query returned no rows')
	}
	mut val := ''
	for _, v in row {
		val = v
		break
	}
	updated_at := get_current_time()
	return val, updated_at
}
```

- [ ] **Step 5: Add /metrics page route**

After the existing `/settings` route handler, add:

```v
@['/metrics']
pub fn (app &App) metrics_page(mut ctx Context) veb.Result {
	return ctx.file('public/index.html')
}
```

- [ ] **Step 6: Add list_metrics handler**

```v
@['/api/metrics'; get]
pub fn (app &App) list_metrics(mut ctx Context) veb.Result {
	rows := app.db.exec("SELECT id, name, slug, metric_type, sql_query, COALESCE(CAST(baseline_value AS TEXT), ''), COALESCE(baseline_updated_at, '') FROM metrics ORDER BY name") or {
		return ctx.server_error('db error')
	}
	mut items := []map[string]string{}
	for r in rows {
		items << {
			'id':                   r.vals[0]
			'name':                 r.vals[1]
			'slug':                 r.vals[2]
			'metric_type':          r.vals[3]
			'sql_query':            r.vals[4]
			'baseline_value':       r.vals[5]
			'baseline_updated_at':  r.vals[6]
		}
	}
	return ctx.json(items)
}
```

- [ ] **Step 7: Add create_metric handler**

```v
@['/api/metrics'; post]
pub fn (app &App) create_metric(mut ctx Context) veb.Result {
	req := json.decode(MetricCreateReq, ctx.req.data) or {
		return ctx.request_error('invalid JSON')
	}
	if req.name.trim_space() == '' {
		return ctx.request_error('name is required')
	}
	if req.slug.trim_space() == '' {
		return ctx.request_error('slug is required')
	}
	existing := app.db.exec("SELECT id FROM metrics WHERE name = '${req.name.trim_space()}' OR slug = '${req.slug.trim_space()}'") or {
		return ctx.server_error('db error')
	}
	if existing.len > 0 {
		ctx.res.set_status(.conflict)
		return ctx.text('name or slug already exists')
	}
	app.db.exec("INSERT INTO metrics (name, slug, metric_type, sql_query) VALUES ('${req.name.trim_space()}', '${req.slug.trim_space()}', '${req.metric_type}', '${req.sql_query.replace("'", "''")}')") or {
		return ctx.server_error('db error')
	}
	rows := app.db.exec("SELECT id, name, slug, metric_type, sql_query, COALESCE(CAST(baseline_value AS TEXT), ''), COALESCE(baseline_updated_at, '') FROM metrics WHERE slug = '${req.slug.trim_space()}'") or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.server_error('insert failed')
	}
	item := {
		'id': rows[0].vals[0], 'name': rows[0].vals[1], 'slug': rows[0].vals[2],
		'metric_type': rows[0].vals[3], 'sql_query': rows[0].vals[4],
		'baseline_value': rows[0].vals[5], 'baseline_updated_at': rows[0].vals[6],
	}
	ctx.res.set_status(.created)
	return ctx.json(item)
}
```

- [ ] **Step 8: Add update_metric handler**

```v
@['/api/metrics/:id'; put]
pub fn (app &App) update_metric(mut ctx Context, id int) veb.Result {
	req := json.decode(MetricUpdateReq, ctx.req.data) or {
		return ctx.request_error('invalid JSON')
	}
	if req.name.trim_space() == '' {
		return ctx.request_error('name is required')
	}
	app.db.exec("UPDATE metrics SET name = '${req.name.trim_space()}', metric_type = '${req.metric_type}', sql_query = '${req.sql_query.replace("'", "''")}' WHERE id = ${id}") or {
		return ctx.server_error('db error')
	}
	rows := app.db.exec("SELECT id, name, slug, metric_type, sql_query, COALESCE(CAST(baseline_value AS TEXT), ''), COALESCE(baseline_updated_at, '') FROM metrics WHERE id = ${id}") or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.not_found()
	}
	return ctx.json({
		'id': rows[0].vals[0], 'name': rows[0].vals[1], 'slug': rows[0].vals[2],
		'metric_type': rows[0].vals[3], 'sql_query': rows[0].vals[4],
		'baseline_value': rows[0].vals[5], 'baseline_updated_at': rows[0].vals[6],
	})
}
```

- [ ] **Step 9: Add delete_metric handler**

```v
@['/api/metrics/:id'; delete]
pub fn (app &App) delete_metric(mut ctx Context, id int) veb.Result {
	app.db.exec('DELETE FROM metrics WHERE id = ${id}') or {
		return ctx.server_error('db error')
	}
	return ctx.text('deleted')
}
```

- [ ] **Step 10: Add refresh_metric handler**

```v
@['/api/metrics/:id/refresh'; post]
pub fn (app &App) refresh_metric(mut ctx Context, id int) veb.Result {
	rows := app.db.exec('SELECT sql_query FROM metrics WHERE id = ${id}') or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.not_found()
	}
	sql_query := rows[0].vals[0]
	value, updated_at := run_metric_sql(sql_query) or {
		ctx.res.set_status(.unprocessable_entity)
		return ctx.json({'error': err.msg()})
	}
	app.db.exec("UPDATE metrics SET baseline_value = ${value}, baseline_updated_at = '${updated_at}' WHERE id = ${id}") or {
		return ctx.server_error('db error')
	}
	return ctx.json({'value': value, 'updated_at': updated_at})
}
```

- [ ] **Step 11: Start the server and verify all endpoints**

```bash
cd apps/tracker
v run .
```

In a separate terminal (using `rtk proxy curl` to bypass RTK filtering):

```bash
# List (empty)
rtk proxy curl -s http://localhost:8080/api/metrics
# Expected: []

# Create
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/metrics \
  -H 'Content-Type: application/json' \
  -d '{"name":"Test CVR","slug":"test_cvr","metric_type":"proportion","sql_query":"SELECT 0.041"}'
# Expected: HTTP:201 with item JSON

# List again
rtk proxy curl -s http://localhost:8080/api/metrics
# Expected: array with one item

# Duplicate name → 409
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/metrics \
  -H 'Content-Type: application/json' \
  -d '{"name":"Test CVR","slug":"test_cvr2","metric_type":"proportion","sql_query":"SELECT 1"}'
# Expected: HTTP:409

# Refresh (simple single-statement SQL)
ID=$(rtk proxy curl -s http://localhost:8080/api/metrics | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/metrics/${ID}/refresh
# Expected: HTTP:200 {"value":"0.041","updated_at":"..."}

# Update
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X PUT http://localhost:8080/api/metrics/${ID} \
  -H 'Content-Type: application/json' \
  -d '{"name":"Checkout CVR","metric_type":"proportion","sql_query":"SELECT 0.041"}'
# Expected: HTTP:200 with updated name

# Delete
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X DELETE http://localhost:8080/api/metrics/${ID}
# Expected: HTTP:200 "deleted"

# /metrics page
rtk proxy curl -s -o /dev/null -w "HTTP:%{http_code}" http://localhost:8080/metrics
# Expected: HTTP:200
```

---

### Task 2: SPA routing — BarChart icon, store wiring, board guard, header icon

**Files:**
- Modify: `apps/tracker/public/app.js` — add `BarChart` icon, `metrics`/`isMetrics`/`loadMetrics` to store, call `loadMetrics()` in init
- Modify: `apps/tracker/public/index.html` — add metrics icon to header, extend board `x-if` guard

**Interfaces:**
- Consumes: Task 1's `GET /api/metrics` endpoint
- Produces:
  - `$store.app.metrics` — array of metric objects available to all components
  - `$store.app.isMetrics` — true when on `/metrics`
  - `$store.app.loadMetrics()` — async function that refreshes the store array
  - `$store.app.Icons.BarChart()` — SVG string

- [ ] **Step 1: Add BarChart icon to the Icons object in app.js**

Find the `Icons` object (around line 208) and add `BarChart` before the closing `};`:

```js
BarChart: () => '<svg width="15" height="15" viewBox="0 0 15 15" fill="none"><rect x="1" y="8" width="3" height="6" rx="1" fill="currentColor" opacity="0.7"/><rect x="6" y="4" width="3" height="10" rx="1" fill="currentColor" opacity="0.9"/><rect x="11" y="1" width="3" height="13" rx="1" fill="currentColor"/></svg>',
```

- [ ] **Step 2: Add metrics state and methods to the Alpine store in app.js**

Find the store's `flaggingSystems: [],` line (around line 239) and add `metrics` below it:

```js
metrics: [],
```

After `get isSettings() { ... },` add:

```js
get isMetrics() { return window.location.pathname === '/metrics'; },
```

After the `loadSettings()` method, add `loadMetrics()`:

```js
async loadMetrics() {
    const data = await fetch('/api/metrics').then(r => r.json());
    this.metrics = data;
},
```

- [ ] **Step 3: Call loadMetrics() in store init()**

In the store's `init()` method, after `this.loadSettings();` add:

```js
this.loadMetrics();
```

- [ ] **Step 4: Extend the board x-if guard in index.html**

Find (around line 99):
```html
<template x-if="!$store.app.isSettings">
```

Replace with:
```html
<template x-if="!$store.app.isSettings && !$store.app.isMetrics">
```

- [ ] **Step 5: Add metrics icon to the header in index.html**

Find (around line 89):
```html
			<a href="/settings" class="btn-settings" title="Settings">
				<span x-html="$store.app.Icons.Gear()"></span>
			</a>
```

Add the metrics icon directly before it:
```html
			<a href="/metrics" class="btn-metrics" title="Metrics catalog">
				<span x-html="$store.app.Icons.BarChart()"></span>
			</a>
```

- [ ] **Step 6: Add .btn-metrics CSS to app.css**

Append to `apps/tracker/public/app.css`:

```css
.btn-metrics { display:flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:8px; color:#6B7280; text-decoration:none; transition:background 0.15s,color 0.15s; }
.btn-metrics:hover { background:#F3F4F6; color:#111827; }
```

- [ ] **Step 7: Restart the server and verify**

Stop and restart the tracker server (`v run .` from `apps/tracker/`).

Verify in browser or curl:
- `GET /` — board still loads, metrics icon appears in header left of gear
- `GET /metrics` — page loads (will be blank until Task 3 adds the template; no JS errors)
- `$store.app.metrics` — open browser console on `/`, run `Alpine.store('app').metrics` — should be an array (empty if no metrics exist yet)
- Board guard — navigating to `/metrics` hides the board content

---

### Task 3: Metrics page UI — CSS, Alpine component, HTML

**Files:**
- Modify: `apps/tracker/public/app.css` — append all `.metrics-*` classes
- Modify: `apps/tracker/public/app.js` — append `Alpine.data('metricsPage', ...)` before closing `});`
- Modify: `apps/tracker/public/index.html` — append `<template x-if="$store.app.isMetrics">` block after the settings `</template>`

**Interfaces:**
- Consumes:
  - `$store.app.metrics` (array), `$store.app.loadMetrics()`, `$store.app.Icons.ArrowLeft()`, `$store.app.isMetrics`
  - `GET/POST/PUT/DELETE /api/metrics`, `POST /api/metrics/:id/refresh`
- Produces: fully functional metrics CRUD page at `/metrics`

- [ ] **Step 1: Append metrics CSS to app.css**

Append to the end of `apps/tracker/public/app.css`:

```css
/* Metrics catalog page */
.metrics-page { max-width:780px; margin:0 auto; padding:40px 24px; }
.metrics-back { display:inline-flex; align-items:center; gap:6px; font-size:13px; color:#6B7280; text-decoration:none; margin-bottom:28px; }
.metrics-back:hover { color:#111827; }
.metrics-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; }
.metrics-title { font-size:20px; font-weight:700; color:#111827; }
.metrics-add-btn { padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; border:none; background:#111827; color:#fff; }
.metrics-add-btn:hover { background:#1F2937; }
.metrics-list { border:1.5px solid #E5E7EB; border-radius:10px; overflow:hidden; }
.metrics-row { display:flex; align-items:center; gap:12px; padding:12px 16px; border-bottom:1px solid #F3F4F6; background:#fff; }
.metrics-row:last-child { border-bottom:none; }
.metrics-row-name { flex:1; font-size:13px; font-weight:600; color:#111827; }
.metrics-row-type { font-size:11px; font-weight:600; padding:2px 8px; border-radius:20px; background:#EFF6FF; color:#1D4ED8; text-transform:uppercase; letter-spacing:0.04em; white-space:nowrap; }
.metrics-row-type.continuous { background:#F0FDF4; color:#15803D; }
.metrics-row-baseline { font-size:13px; color:#374151; min-width:60px; text-align:right; font-variant-numeric:tabular-nums; }
.metrics-row-age { font-size:11px; color:#9CA3AF; min-width:70px; text-align:right; }
.metrics-row-actions { display:flex; align-items:center; gap:6px; }
.metrics-btn-refresh { padding:4px 8px; border-radius:6px; font-size:13px; font-weight:500; cursor:pointer; border:none; background:#EFF6FF; color:#1D4ED8; }
.metrics-btn-refresh:hover { background:#DBEAFE; }
.metrics-btn-refresh:disabled { opacity:0.5; cursor:default; }
.metrics-btn-edit { padding:4px 10px; border-radius:6px; font-size:12px; font-weight:500; cursor:pointer; border:none; background:#F3F4F6; color:#374151; }
.metrics-btn-edit:hover { background:#E5E7EB; }
.metrics-btn-delete { padding:4px 10px; border-radius:6px; font-size:12px; font-weight:500; cursor:pointer; border:none; background:#FEF2F2; color:#DC2626; }
.metrics-btn-delete:hover { background:#FEE2E2; }
.metrics-empty { padding:32px; text-align:center; color:#9CA3AF; font-size:13px; }
.metrics-modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.4); z-index:1000; display:flex; align-items:center; justify-content:center; }
.metrics-modal-box { background:#fff; border-radius:14px; padding:28px; width:600px; max-width:calc(100vw - 32px); max-height:calc(100vh - 48px); overflow-y:auto; box-shadow:0 20px 60px rgba(0,0,0,0.18); }
.metrics-modal-title { font-size:17px; font-weight:700; color:#111827; margin-bottom:24px; }
.metrics-modal-field { margin-bottom:16px; }
.metrics-modal-label { display:block; font-size:12px; font-weight:600; color:#374151; margin-bottom:6px; }
.metrics-modal-input { width:100%; padding:8px 12px; border:1.5px solid #E5E7EB; border-radius:8px; font-size:13px; color:#111827; outline:none; font-family:inherit; }
.metrics-modal-input:focus { border-color:#3B82F6; }
.metrics-modal-input:disabled { background:#F9FAFB; color:#9CA3AF; cursor:default; }
.metrics-modal-select { width:100%; padding:8px 12px; border:1.5px solid #E5E7EB; border-radius:8px; font-size:13px; color:#111827; outline:none; background:#fff; font-family:inherit; }
.metrics-modal-select:focus { border-color:#3B82F6; }
.metrics-modal-textarea { width:100%; padding:8px 12px; border:1.5px solid #E5E7EB; border-radius:8px; font-size:12px; color:#111827; outline:none; font-family:'Courier New',monospace; resize:vertical; }
.metrics-modal-textarea:focus { border-color:#3B82F6; }
.metrics-modal-footer { display:flex; justify-content:flex-end; gap:10px; margin-top:24px; }
.metrics-btn-cancel { padding:8px 18px; border-radius:8px; font-size:13px; font-weight:500; cursor:pointer; border:1.5px solid #E5E7EB; background:#fff; color:#374151; }
.metrics-btn-cancel:hover { background:#F3F4F6; }
.metrics-btn-save { padding:8px 18px; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; border:none; background:#111827; color:#fff; }
.metrics-btn-save:hover { background:#1F2937; }
```

- [ ] **Step 2: Append metricsPage Alpine.data component to app.js**

Find the closing `});` of the `document.addEventListener('alpine:init', ...)` block (currently the last line of the file, after the `settingsPage` component). Insert the following BEFORE that closing `});`:

```js
Alpine.data('metricsPage', () => ({
    get metrics() { return Alpine.store('app').metrics; },
    showModal: false,
    editing: null,
    form: { name: '', slug: '', metric_type: 'proportion', sql_query: '' },
    refreshing: {},

    slugify(name) {
        return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    },

    onNameInput() {
        if (this.editing === null) {
            this.form.slug = this.slugify(this.form.name);
        }
    },

    openNew() {
        this.editing = null;
        this.form = { name: '', slug: '', metric_type: 'proportion', sql_query: '' };
        this.showModal = true;
    },

    openEdit(metric) {
        this.editing = metric.id;
        this.form = {
            name: metric.name,
            slug: metric.slug,
            metric_type: metric.metric_type,
            sql_query: metric.sql_query,
        };
        this.showModal = true;
    },

    closeModal() {
        this.showModal = false;
        this.editing = null;
    },

    async save() {
        if (!this.form.name.trim() || !this.form.slug.trim()) return;
        const url = this.editing ? `/api/metrics/${this.editing}` : '/api/metrics';
        const method = this.editing ? 'PUT' : 'POST';
        const body = this.editing
            ? { name: this.form.name, metric_type: this.form.metric_type, sql_query: this.form.sql_query }
            : { name: this.form.name, slug: this.form.slug, metric_type: this.form.metric_type, sql_query: this.form.sql_query };
        const resp = await fetch(url, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (resp.ok) {
            this.closeModal();
            await Alpine.store('app').loadMetrics();
        } else {
            alert(resp.status === 409 ? 'Name or slug already exists.' : 'Save failed.');
        }
    },

    async deleteMetric(id, name) {
        if (!confirm(`Delete metric "${name}"?`)) return;
        await fetch(`/api/metrics/${id}`, { method: 'DELETE' });
        await Alpine.store('app').loadMetrics();
    },

    async refresh(id) {
        this.refreshing = { ...this.refreshing, [id]: true };
        const resp = await fetch(`/api/metrics/${id}/refresh`, { method: 'POST' });
        this.refreshing = { ...this.refreshing, [id]: false };
        if (resp.ok) {
            await Alpine.store('app').loadMetrics();
        } else {
            const data = await resp.json().catch(() => ({}));
            alert('Refresh failed: ' + (data.error || 'unknown error'));
        }
    },

    formatAge(updated_at) {
        if (!updated_at) return 'never';
        const d = new Date(updated_at.replace(' ', 'T'));
        const diff = Math.floor((Date.now() - d) / 1000);
        if (diff < 60) return 'just now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        return Math.floor(diff / 86400) + 'd ago';
    },
}));
```

- [ ] **Step 3: Add metrics page HTML to index.html**

Find the closing `</template>` of the settings block (the one that wraps the `<div class="settings-page"...>`). It is just before `</body>`. After that `</template>`, insert:

```html
<template x-if="$store.app.isMetrics">
<div class="metrics-page" x-data="metricsPage()">
    <a href="/" class="metrics-back">
        <span x-html="$store.app.Icons.ArrowLeft()"></span> Board
    </a>
    <div class="metrics-header">
        <div class="metrics-title">Metrics Catalog</div>
        <button class="metrics-add-btn" @click="openNew()">+ New metric</button>
    </div>

    <template x-if="metrics.length === 0">
        <div class="metrics-empty">No metrics yet. Click "New metric" to add one.</div>
    </template>

    <template x-if="metrics.length > 0">
        <div class="metrics-list">
            <template x-for="m in metrics" :key="m.id">
                <div class="metrics-row">
                    <span class="metrics-row-name" x-text="m.name"></span>
                    <span class="metrics-row-type" :class="{ continuous: m.metric_type === 'continuous' }" x-text="m.metric_type"></span>
                    <span class="metrics-row-baseline" x-text="m.baseline_value ? Number(m.baseline_value).toPrecision(4) : '—'"></span>
                    <span class="metrics-row-age" x-text="formatAge(m.baseline_updated_at)"></span>
                    <div class="metrics-row-actions">
                        <button class="metrics-btn-refresh" :disabled="refreshing[m.id]"
                            @click="refresh(m.id)" title="Refresh baseline">↻</button>
                        <button class="metrics-btn-edit" @click="openEdit(m)">Edit</button>
                        <button class="metrics-btn-delete" @click="deleteMetric(m.id, m.name)">Delete</button>
                    </div>
                </div>
            </template>
        </div>
    </template>

    <template x-if="showModal">
        <div class="metrics-modal-overlay" @click.self="closeModal()">
            <div class="metrics-modal-box">
                <div class="metrics-modal-title" x-text="editing ? 'Edit metric' : 'New metric'"></div>
                <div class="metrics-modal-field">
                    <label class="metrics-modal-label">Name</label>
                    <input class="metrics-modal-input" x-model="form.name" @input="onNameInput()"
                        placeholder="Checkout CVR">
                </div>
                <div class="metrics-modal-field">
                    <label class="metrics-modal-label">Slug</label>
                    <input class="metrics-modal-input" x-model="form.slug" :disabled="editing !== null"
                        placeholder="checkout_cvr">
                </div>
                <div class="metrics-modal-field">
                    <label class="metrics-modal-label">Type</label>
                    <select class="metrics-modal-select" x-model="form.metric_type">
                        <option value="proportion">Proportion (rate 0–1, e.g. CVR, CTR)</option>
                        <option value="continuous">Continuous (e.g. revenue, latency)</option>
                    </select>
                </div>
                <div class="metrics-modal-field">
                    <label class="metrics-modal-label">SQL query</label>
                    <textarea class="metrics-modal-textarea" rows="12" x-model="form.sql_query"
                        placeholder="INSTALL httpfs;&#10;LOAD httpfs;&#10;-- set secrets here&#10;SELECT COUNT(CASE WHEN converted THEN 1 END)::FLOAT / COUNT(*) FROM events WHERE date >= current_date - 30"></textarea>
                </div>
                <div class="metrics-modal-footer">
                    <button class="metrics-btn-cancel" @click="closeModal()">Cancel</button>
                    <button class="metrics-btn-save" @click="save()">Save</button>
                </div>
            </div>
        </div>
    </template>
</div>
</template>
```

- [ ] **Step 4: Restart server and verify full UI flow**

Stop and restart (`v run .` from `apps/tracker/`). Open `http://localhost:8080/metrics` in the browser and verify:

1. **Empty state** — "No metrics yet" message displays
2. **Create** — click "+ New metric", fill Name (slug auto-fills), select type, enter `SELECT 0.05` as SQL, click Save → metric appears in list
3. **Refresh** — click ↻ → baseline updates to `0.05000`, timestamp shows "just now"
4. **Edit** — click Edit → modal opens with slug field disabled, change name, Save → list updates
5. **Type badge color** — proportion = blue badge, continuous = green badge
6. **Delete** — confirm dialog, metric removed
7. **Board navigation** — clicking "Board" in header returns to `/` board; experiments still load correctly
8. **No console errors** — open DevTools, check for Alpine or fetch errors

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `metrics` SQLite table with all 7 columns | Task 1 Step 3 |
| `metric_type` = 'proportion' or 'continuous' | Task 1 Step 7 (stored as-is, select enforces values in UI) |
| Slug auto-generated, immutable after create | Task 3 Step 2 (`slugify` + `editing !== null` disables field); PUT ignores slug |
| GET /api/metrics ordered by name | Task 1 Step 6 |
| POST 400 on blank name/slug | Task 1 Step 7 |
| POST/PUT 409 on duplicate (pre-insert SELECT) | Task 1 Step 7 |
| PUT ignores slug field | Task 1 Step 8 (MetricUpdateReq has no slug field) |
| DELETE 200 always | Task 1 Step 9 |
| Refresh: split on `;`, run each stmt, get_first_row | Task 1 Step 4 |
| Refresh: 422 `{"error":"..."}` on SQL failure | Task 1 Step 10 |
| Refresh: stores baseline_value + baseline_updated_at | Task 1 Step 10 |
| v.mod adds vduckdb dependency | Task 1 Step 1 |
| GET /metrics serves index.html | Task 1 Step 5 |
| isMetrics getter in store | Task 2 Step 2 |
| loadMetrics() in store init | Task 2 Step 3 |
| Board x-if extended to exclude metrics | Task 2 Step 4 |
| BarChart icon in header linking to /metrics | Task 2 Steps 1+5 |
| .btn-metrics CSS | Task 2 Step 6 |
| metricsPage Alpine.data with all 7 methods | Task 3 Step 2 |
| Modal for create/edit | Task 3 Step 3 |
| Slug auto-fills from name on openNew | Task 3 Step 2 (`onNameInput`) |
| Slug disabled on edit | Task 3 Step 3 (`:disabled="editing !== null"`) |
| All .metrics-* CSS classes from spec | Task 3 Step 1 |
| formatAge helper | Task 3 Step 2 |
| Refresh button per-row with loading state | Task 3 Steps 2+3 |
| Empty state message | Task 3 Step 3 |
| Baseline value displayed to 4 significant figures | Task 3 Step 3 (`toPrecision(4)`) |
