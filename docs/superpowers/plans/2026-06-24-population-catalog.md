# Population Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/populations` catalog page to the tracker where teams manage reusable eligible-population definitions (name, slug, DuckDB SQL returning a count) and refresh their counts on demand.

**Architecture:** SQLite stores population definitions and cached counts; the existing `run_metric_sql` helper (already in `main.v`) executes the multi-statement SQL on refresh and writes the scalar count back to SQLite. The Alpine SPA adds a `populationsPage` component and store wiring mirroring the existing `/metrics` and `/settings` patterns.

**Tech Stack:** V + veb (backend), SQLite (storage), vduckdb via existing `run_metric_sql` helper (count execution), Alpine.js 3 (frontend), vanilla CSS

## Global Constraints

- All backend code lives in `apps/tracker/main.v` — do not create new V files
- All frontend changes are in `apps/tracker/public/app.js`, `apps/tracker/public/index.html`, `apps/tracker/public/app.css`
- `apps/` is git-ignored — no commits; verify by curl/browser instead
- Reuse the existing `run_metric_sql(sql_query string) !(string, string)` helper — do NOT create a new DuckDB helper function
- `vduckdb` is already imported — no v.mod change needed
- Duplicate detection uses pre-insert SELECT (not relying on UNIQUE constraint)
- Slug is immutable after creation — PUT endpoint uses `PopulationUpdateReq` which has no slug field
- Refresh returns key `count` (not `value`) — `{"count": "84320", "updated_at": "..."}`
- Server runs on port 8080; use `rtk proxy curl` for all HTTP testing

---

### Task 1: Backend — populations table + 5 API endpoints + /populations route

**Files:**
- Modify: `apps/tracker/main.v`

**Interfaces:**
- Produces:
  - `GET /api/populations` → `[{"id","name","slug","sql_query","population_count","count_updated_at"}]`
  - `POST /api/populations` body `{"name","slug","sql_query"}` → 201 item or 400/409
  - `PUT /api/populations/:id` body `{"name","sql_query"}` → 200 item or 400/404
  - `DELETE /api/populations/:id` → 200 "deleted"
  - `POST /api/populations/:id/refresh` → 200 `{"count","updated_at"}` or 422 `{"error":"..."}`
  - `GET /populations` → 200 (serves index.html)

- [ ] **Step 1: Add PopulationCreateReq and PopulationUpdateReq structs**

Find the `MetricUpdateReq` struct (around line 627 — search for it). Add the two population structs immediately after it:

```v
struct PopulationCreateReq {
	name      string
	slug      string
	sql_query string
}

struct PopulationUpdateReq {
	name      string
	sql_query string
}
```

- [ ] **Step 2: Add populations migration**

Find the `run_migrations()` function. Its `migrations` array ends with the metrics `CREATE TABLE` line (line ~694). Append the populations table entry as the next element:

```v
"CREATE TABLE IF NOT EXISTS populations (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, slug TEXT NOT NULL UNIQUE, sql_query TEXT NOT NULL DEFAULT '', population_count REAL, count_updated_at TEXT)",
```

- [ ] **Step 3: Add /populations page route**

Find the `/metrics` route handler (around line 82). Add the populations route immediately after it:

```v
@['/populations']
pub fn (app &App) populations_page(mut ctx Context) veb.Result {
	return ctx.file('public/index.html')
}
```

- [ ] **Step 4: Add list_populations handler**

Add after the last existing API handler (or after the metrics handlers block — search for `@['/api/metrics/:id/refresh'; post]` to find the end):

```v
@['/api/populations'; get]
pub fn (app &App) list_populations(mut ctx Context) veb.Result {
	rows := app.db.exec("SELECT id, name, slug, sql_query, COALESCE(CAST(population_count AS TEXT), ''), COALESCE(count_updated_at, '') FROM populations ORDER BY name") or {
		return ctx.server_error('db error')
	}
	mut items := []map[string]string{}
	for r in rows {
		items << {
			'id':               r.vals[0]
			'name':             r.vals[1]
			'slug':             r.vals[2]
			'sql_query':        r.vals[3]
			'population_count': r.vals[4]
			'count_updated_at': r.vals[5]
		}
	}
	return ctx.json(items)
}
```

- [ ] **Step 5: Add create_population handler**

```v
@['/api/populations'; post]
pub fn (app &App) create_population(mut ctx Context) veb.Result {
	req := json.decode(PopulationCreateReq, ctx.req.data) or {
		return ctx.request_error('invalid JSON')
	}
	if req.name.trim_space() == '' {
		return ctx.request_error('name is required')
	}
	if req.slug.trim_space() == '' {
		return ctx.request_error('slug is required')
	}
	existing := app.db.exec("SELECT id FROM populations WHERE name = '${req.name.trim_space()}' OR slug = '${req.slug.trim_space()}'") or {
		return ctx.server_error('db error')
	}
	if existing.len > 0 {
		ctx.res.set_status(.conflict)
		return ctx.text('name or slug already exists')
	}
	app.db.exec("INSERT INTO populations (name, slug, sql_query) VALUES ('${req.name.trim_space()}', '${req.slug.trim_space()}', '${req.sql_query.replace("'", "''")}')") or {
		return ctx.server_error('db error')
	}
	rows := app.db.exec("SELECT id, name, slug, sql_query, COALESCE(CAST(population_count AS TEXT), ''), COALESCE(count_updated_at, '') FROM populations WHERE slug = '${req.slug.trim_space()}'") or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.server_error('insert failed')
	}
	item := {
		'id': rows[0].vals[0], 'name': rows[0].vals[1], 'slug': rows[0].vals[2],
		'sql_query': rows[0].vals[3], 'population_count': rows[0].vals[4],
		'count_updated_at': rows[0].vals[5],
	}
	ctx.res.set_status(.created)
	return ctx.json(item)
}
```

- [ ] **Step 6: Add update_population handler**

```v
@['/api/populations/:id'; put]
pub fn (app &App) update_population(mut ctx Context, id int) veb.Result {
	req := json.decode(PopulationUpdateReq, ctx.req.data) or {
		return ctx.request_error('invalid JSON')
	}
	if req.name.trim_space() == '' {
		return ctx.request_error('name is required')
	}
	app.db.exec("UPDATE populations SET name = '${req.name.trim_space()}', sql_query = '${req.sql_query.replace("'", "''")}' WHERE id = ${id}") or {
		return ctx.server_error('db error')
	}
	rows := app.db.exec("SELECT id, name, slug, sql_query, COALESCE(CAST(population_count AS TEXT), ''), COALESCE(count_updated_at, '') FROM populations WHERE id = ${id}") or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.not_found()
	}
	return ctx.json({
		'id': rows[0].vals[0], 'name': rows[0].vals[1], 'slug': rows[0].vals[2],
		'sql_query': rows[0].vals[3], 'population_count': rows[0].vals[4],
		'count_updated_at': rows[0].vals[5],
	})
}
```

- [ ] **Step 7: Add delete_population handler**

```v
@['/api/populations/:id'; delete]
pub fn (app &App) delete_population(mut ctx Context, id int) veb.Result {
	app.db.exec('DELETE FROM populations WHERE id = ${id}') or {
		return ctx.server_error('db error')
	}
	return ctx.text('deleted')
}
```

- [ ] **Step 8: Add refresh_population handler**

```v
@['/api/populations/:id/refresh'; post]
pub fn (app &App) refresh_population(mut ctx Context, id int) veb.Result {
	rows := app.db.exec('SELECT sql_query FROM populations WHERE id = ${id}') or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.not_found()
	}
	sql_query := rows[0].vals[0]
	count, updated_at := run_metric_sql(sql_query) or {
		ctx.res.set_status(.unprocessable_entity)
		return ctx.json({'error': err.msg()})
	}
	app.db.exec("UPDATE populations SET population_count = ${count}, count_updated_at = '${updated_at}' WHERE id = ${id}") or {
		return ctx.server_error('db error')
	}
	return ctx.json({'count': count, 'updated_at': updated_at})
}
```

- [ ] **Step 9: Restart the server and verify all endpoints**

Stop any running server instance, then:

```bash
cd /home/rabt/devel/vstats/apps/tracker
fuser -k 8080/tcp 2>/dev/null; sleep 1
v run . &
sleep 5
```

Run these curl tests (using `rtk proxy curl` to bypass output filtering):

```bash
# List (empty)
rtk proxy curl -s http://localhost:8080/api/populations
# Expected: []

# Create
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/populations \
  -H 'Content-Type: application/json' \
  -d '{"name":"All active users","slug":"all_active_users","sql_query":"SELECT 50000"}'
# Expected: HTTP:201 with item JSON

# Duplicate → 409
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/populations \
  -H 'Content-Type: application/json' \
  -d '{"name":"All active users","slug":"all_active_users2","sql_query":"SELECT 1"}'
# Expected: HTTP:409

# Refresh
ID=$(rtk proxy curl -s http://localhost:8080/api/populations | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/populations/${ID}/refresh
# Expected: HTTP:200 {"count":"50000","updated_at":"..."}

# Update
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X PUT http://localhost:8080/api/populations/${ID} \
  -H 'Content-Type: application/json' \
  -d '{"name":"All active users (30d)","sql_query":"SELECT 50000"}'
# Expected: HTTP:200 with updated name

# Delete
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X DELETE http://localhost:8080/api/populations/${ID}
# Expected: HTTP:200 "deleted"

# Page route
rtk proxy curl -s -o /dev/null -w "HTTP:%{http_code}" http://localhost:8080/populations
# Expected: HTTP:200
```

---

### Task 2: SPA routing — Users icon, store wiring, board guard, header icon

**Files:**
- Modify: `apps/tracker/public/app.js`
- Modify: `apps/tracker/public/index.html`
- Modify: `apps/tracker/public/app.css`

**Interfaces:**
- Consumes: Task 1's `GET /api/populations` endpoint
- Produces:
  - `$store.app.populations` — array of population objects
  - `$store.app.isPopulations` — true when on `/populations`
  - `$store.app.loadPopulations()` — async refresh function
  - `$store.app.Icons.Users()` — SVG string

- [ ] **Step 1: Add Users icon to the Icons object in app.js**

Find the `Icons` object (around line 208 — search for `const Icons = {`). Add `Users` before the closing `};`:

```js
Users: () => '<svg width="15" height="15" viewBox="0 0 15 15" fill="none"><circle cx="5.5" cy="4.5" r="2.5" stroke="currentColor" stroke-width="1.3"/><path d="M1 13c0-2.5 2-4 4.5-4s4.5 1.5 4.5 4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><circle cx="11" cy="5" r="2" stroke="currentColor" stroke-width="1.3"/><path d="M13 13c0-2-1.3-3.2-3-3.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>',
```

- [ ] **Step 2: Add populations state and methods to the Alpine store**

Find `metrics: [],` in the store (around line 240). Add `populations` directly below it:

```js
populations: [],
```

Find `get isMetrics() { ... },` (line ~245). Add `isPopulations` directly after it:

```js
get isPopulations() { return window.location.pathname === '/populations'; },
```

Find `async loadMetrics() { ... },` (lines ~256-259). Add `loadPopulations` directly after it:

```js
async loadPopulations() {
    const data = await fetch('/api/populations').then(r => r.json());
    this.populations = data;
},
```

- [ ] **Step 3: Call loadPopulations() in store init()**

Find `this.loadMetrics();` (line ~329). Add the populations call on the next line:

```js
this.loadPopulations();
```

- [ ] **Step 4: Extend the board x-if guard in index.html**

Find (line ~102):
```html
<template x-if="!$store.app.isSettings && !$store.app.isMetrics">
```

Replace with:
```html
<template x-if="!$store.app.isSettings && !$store.app.isMetrics && !$store.app.isPopulations">
```

- [ ] **Step 5: Add populations icon to the header in index.html**

Find (line ~89):
```html
			<a href="/metrics" class="btn-metrics" title="Metrics catalog">
				<span x-html="$store.app.Icons.BarChart()"></span>
			</a>
```

Add the populations icon directly before it:
```html
			<a href="/populations" class="btn-populations" title="Populations">
				<span x-html="$store.app.Icons.Users()"></span>
			</a>
```

- [ ] **Step 6: Add .btn-populations CSS to app.css**

Append to the end of `apps/tracker/public/app.css`:

```css
.btn-populations { display:flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:8px; color:#6B7280; text-decoration:none; transition:background 0.15s,color 0.15s; }
.btn-populations:hover { background:#F3F4F6; color:#111827; }
```

- [ ] **Step 7: Restart the server and verify**

```bash
fuser -k 8080/tcp 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . &
sleep 5
```

Verify:
```bash
# /populations route
rtk proxy curl -s -o /dev/null -w "HTTP:%{http_code}" http://localhost:8080/populations
# Expected: HTTP:200

# isPopulations getter present in JS
rtk proxy curl -s http://localhost:8080/app.js | grep -c "isPopulations"
# Expected: 1

# Board guard updated
rtk proxy curl -s http://localhost:8080/index.html 2>/dev/null || rtk proxy curl -s http://localhost:8080/ | grep -c "isPopulations"
# Expected: 1 (guard contains isPopulations)
```

---

### Task 3: Populations page UI — CSS, Alpine component, HTML

**Files:**
- Modify: `apps/tracker/public/app.css` — append all `.populations-*` classes
- Modify: `apps/tracker/public/app.js` — append `Alpine.data('populationsPage', ...)` before closing `});`
- Modify: `apps/tracker/public/index.html` — append `<template x-if="$store.app.isPopulations">` block after the metrics `</template>`

**Interfaces:**
- Consumes:
  - `$store.app.populations` (array), `$store.app.loadPopulations()`, `$store.app.Icons.ArrowLeft()`, `$store.app.isPopulations`
  - `GET/POST/PUT/DELETE /api/populations`, `POST /api/populations/:id/refresh`
- Produces: fully functional populations CRUD page at `/populations`

- [ ] **Step 1: Append populations CSS to app.css**

Append to the end of `apps/tracker/public/app.css` (after the `.btn-populations` lines added in Task 2):

```css
/* Populations catalog page */
.populations-page { max-width:780px; margin:0 auto; padding:40px 24px; }
.populations-back { display:inline-flex; align-items:center; gap:6px; font-size:13px; color:#6B7280; text-decoration:none; margin-bottom:28px; }
.populations-back:hover { color:#111827; }
.populations-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; }
.populations-title { font-size:20px; font-weight:700; color:#111827; }
.populations-add-btn { padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; border:none; background:#111827; color:#fff; }
.populations-add-btn:hover { background:#1F2937; }
.populations-list { border:1.5px solid #E5E7EB; border-radius:10px; overflow:hidden; }
.populations-row { display:flex; align-items:center; gap:12px; padding:12px 16px; border-bottom:1px solid #F3F4F6; background:#fff; }
.populations-row:last-child { border-bottom:none; }
.populations-row-name { flex:1; font-size:13px; font-weight:600; color:#111827; }
.populations-row-count { font-size:13px; color:#374151; min-width:80px; text-align:right; font-variant-numeric:tabular-nums; }
.populations-row-age { font-size:11px; color:#9CA3AF; min-width:70px; text-align:right; }
.populations-row-actions { display:flex; align-items:center; gap:6px; }
.populations-btn-refresh { padding:4px 8px; border-radius:6px; font-size:13px; font-weight:500; cursor:pointer; border:none; background:#EFF6FF; color:#1D4ED8; }
.populations-btn-refresh:hover { background:#DBEAFE; }
.populations-btn-refresh:disabled { opacity:0.5; cursor:default; }
.populations-btn-edit { padding:4px 10px; border-radius:6px; font-size:12px; font-weight:500; cursor:pointer; border:none; background:#F3F4F6; color:#374151; }
.populations-btn-edit:hover { background:#E5E7EB; }
.populations-btn-delete { padding:4px 10px; border-radius:6px; font-size:12px; font-weight:500; cursor:pointer; border:none; background:#FEF2F2; color:#DC2626; }
.populations-btn-delete:hover { background:#FEE2E2; }
.populations-empty { padding:32px; text-align:center; color:#9CA3AF; font-size:13px; }
.populations-modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.4); z-index:1000; display:flex; align-items:center; justify-content:center; }
.populations-modal-box { background:#fff; border-radius:14px; padding:28px; width:600px; max-width:calc(100vw - 32px); max-height:calc(100vh - 48px); overflow-y:auto; box-shadow:0 20px 60px rgba(0,0,0,0.18); }
.populations-modal-title { font-size:17px; font-weight:700; color:#111827; margin-bottom:24px; }
.populations-modal-field { margin-bottom:16px; }
.populations-modal-label { display:block; font-size:12px; font-weight:600; color:#374151; margin-bottom:6px; }
.populations-modal-input { width:100%; padding:8px 12px; border:1.5px solid #E5E7EB; border-radius:8px; font-size:13px; color:#111827; outline:none; font-family:inherit; }
.populations-modal-input:focus { border-color:#3B82F6; }
.populations-modal-input:disabled { background:#F9FAFB; color:#9CA3AF; cursor:default; }
.populations-modal-textarea { width:100%; padding:8px 12px; border:1.5px solid #E5E7EB; border-radius:8px; font-size:12px; color:#111827; outline:none; font-family:'Courier New',monospace; resize:vertical; }
.populations-modal-textarea:focus { border-color:#3B82F6; }
.populations-modal-footer { display:flex; justify-content:flex-end; gap:10px; margin-top:24px; }
.populations-btn-cancel { padding:8px 18px; border-radius:8px; font-size:13px; font-weight:500; cursor:pointer; border:1.5px solid #E5E7EB; background:#fff; color:#374151; }
.populations-btn-cancel:hover { background:#F3F4F6; }
.populations-btn-save { padding:8px 18px; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; border:none; background:#111827; color:#fff; }
.populations-btn-save:hover { background:#1F2937; }
```

- [ ] **Step 2: Append populationsPage Alpine.data component to app.js**

Find the closing `});` of the `document.addEventListener('alpine:init', ...)` block — it is the last line of the file (currently after the `metricsPage` component closing `}));`). Insert the following BEFORE that closing `});`:

```js
Alpine.data('populationsPage', () => ({
    get populations() { return Alpine.store('app').populations; },
    showModal: false,
    editing: null,
    form: { name: '', slug: '', sql_query: '' },
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
        this.form = { name: '', slug: '', sql_query: '' };
        this.showModal = true;
    },

    openEdit(pop) {
        this.editing = pop.id;
        this.form = { name: pop.name, slug: pop.slug, sql_query: pop.sql_query };
        this.showModal = true;
    },

    closeModal() {
        this.showModal = false;
        this.editing = null;
    },

    async save() {
        if (!this.form.name.trim() || !this.form.slug.trim()) return;
        const url = this.editing ? `/api/populations/${this.editing}` : '/api/populations';
        const method = this.editing ? 'PUT' : 'POST';
        const body = this.editing
            ? { name: this.form.name, sql_query: this.form.sql_query }
            : { name: this.form.name, slug: this.form.slug, sql_query: this.form.sql_query };
        const resp = await fetch(url, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (resp.ok) {
            this.closeModal();
            await Alpine.store('app').loadPopulations();
        } else {
            alert(resp.status === 409 ? 'Name or slug already exists.' : 'Save failed.');
        }
    },

    async deletePop(id, name) {
        if (!confirm(`Delete population "${name}"?`)) return;
        await fetch(`/api/populations/${id}`, { method: 'DELETE' });
        await Alpine.store('app').loadPopulations();
    },

    async refresh(id) {
        this.refreshing = { ...this.refreshing, [id]: true };
        const resp = await fetch(`/api/populations/${id}/refresh`, { method: 'POST' });
        this.refreshing = { ...this.refreshing, [id]: false };
        if (resp.ok) {
            await Alpine.store('app').loadPopulations();
        } else {
            const data = await resp.json().catch(() => ({}));
            alert('Refresh failed: ' + (data.error || 'unknown error'));
        }
    },

    formatCount(count) {
        if (!count) return '—';
        return Number(count).toLocaleString();
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

- [ ] **Step 3: Add populations page HTML to index.html**

Find the closing `</template>` of the metrics block (line ~1226 — it is the last `</template>` before `</body></html>`). Insert the following after that `</template>` and before `</body>`:

```html
<template x-if="$store.app.isPopulations">
<div class="populations-page" x-data="populationsPage()">
    <a href="/" class="populations-back">
        <span x-html="$store.app.Icons.ArrowLeft()"></span> Board
    </a>
    <div class="populations-header">
        <div class="populations-title">Populations</div>
        <button class="populations-add-btn" @click="openNew()">+ New population</button>
    </div>

    <template x-if="populations.length === 0">
        <div class="populations-empty">No populations yet. Click "New population" to add one.</div>
    </template>

    <template x-if="populations.length > 0">
        <div class="populations-list">
            <template x-for="p in populations" :key="p.id">
                <div class="populations-row">
                    <span class="populations-row-name" x-text="p.name"></span>
                    <span class="populations-row-count" x-text="formatCount(p.population_count)"></span>
                    <span class="populations-row-age" x-text="formatAge(p.count_updated_at)"></span>
                    <div class="populations-row-actions">
                        <button class="populations-btn-refresh" :disabled="refreshing[p.id]"
                            @click="refresh(p.id)" title="Refresh count">↻</button>
                        <button class="populations-btn-edit" @click="openEdit(p)">Edit</button>
                        <button class="populations-btn-delete" @click="deletePop(p.id, p.name)">Delete</button>
                    </div>
                </div>
            </template>
        </div>
    </template>

    <template x-if="showModal">
        <div class="populations-modal-overlay" @click.self="closeModal()">
            <div class="populations-modal-box">
                <div class="populations-modal-title" x-text="editing ? 'Edit population' : 'New population'"></div>
                <div class="populations-modal-field">
                    <label class="populations-modal-label">Name</label>
                    <input class="populations-modal-input" x-model="form.name" @input="onNameInput()"
                        placeholder="Mobile users — last 30d">
                </div>
                <div class="populations-modal-field">
                    <label class="populations-modal-label">Slug</label>
                    <input class="populations-modal-input" x-model="form.slug" :disabled="editing !== null"
                        placeholder="mobile_users_30d">
                </div>
                <div class="populations-modal-field">
                    <label class="populations-modal-label">SQL query</label>
                    <textarea class="populations-modal-textarea" rows="12" x-model="form.sql_query"
                        placeholder="INSTALL httpfs;&#10;LOAD httpfs;&#10;-- set secrets here&#10;SELECT COUNT(DISTINCT user_id) FROM events WHERE date >= current_date - 30 AND platform = 'mobile'"></textarea>
                </div>
                <div class="populations-modal-footer">
                    <button class="populations-btn-cancel" @click="closeModal()">Cancel</button>
                    <button class="populations-btn-save" @click="save()">Save</button>
                </div>
            </div>
        </div>
    </template>
</div>
</template>
```

- [ ] **Step 4: Restart server and verify full UI flow**

```bash
fuser -k 8080/tcp 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . &
sleep 5
```

Verify:
```bash
# /populations page contains the component
rtk proxy curl -s http://localhost:8080/populations | grep -c "populations-page"
# Expected: 1

# CSS present
rtk proxy curl -s http://localhost:8080/app.css | grep -c "populations-page"
# Expected: 1

# Alpine component present
rtk proxy curl -s http://localhost:8080/app.js | grep -c "populationsPage"
# Expected: 1

# Create + refresh end-to-end
ID=$(rtk proxy curl -s -X POST http://localhost:8080/api/populations \
  -H 'Content-Type: application/json' \
  -d '{"name":"Smoke test pop","slug":"smoke_test_pop","sql_query":"SELECT 99999"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
rtk proxy curl -s -X POST http://localhost:8080/api/populations/${ID}/refresh
# Expected: {"count":"99999","updated_at":"..."}

# Board still loads
rtk proxy curl -s -o /dev/null -w "HTTP:%{http_code}" http://localhost:8080/
# Expected: HTTP:200
```

Also open `http://localhost:8080/populations` in the browser and verify:
1. "No populations yet" empty state displays
2. "+ New population" → modal opens, slug auto-fills from name
3. Save → population appears in list with count "—" and age "never"
4. ↻ → count updates to `99,999`, age shows "just now"
5. Edit → modal opens with slug disabled
6. Delete → confirmation dialog, entry removed
7. Board at `/` still works with no JS errors

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `populations` SQLite table with 6 columns | Task 1 Step 2 |
| GET /api/populations ordered by name | Task 1 Step 4 |
| POST 400 on blank name/slug | Task 1 Step 5 |
| POST/PUT 409 on duplicate (pre-insert SELECT) | Task 1 Step 5 |
| PUT ignores slug field (PopulationUpdateReq has no slug) | Task 1 Step 6 |
| DELETE 200 always | Task 1 Step 7 |
| Refresh reuses run_metric_sql (no new helper) | Task 1 Step 8 |
| Refresh returns key `count` not `value` | Task 1 Step 8 |
| Refresh: 422 `{"error":"..."}` on SQL failure | Task 1 Step 8 |
| Refresh: stores population_count + count_updated_at | Task 1 Step 8 |
| GET /populations serves index.html | Task 1 Step 3 |
| isPopulations getter in store | Task 2 Step 2 |
| loadPopulations() in store init | Task 2 Step 3 |
| Board x-if extended to exclude populations | Task 2 Step 4 |
| Users SVG icon in header linking to /populations | Task 2 Steps 1+5 |
| .btn-populations CSS | Task 2 Step 6 |
| populationsPage Alpine.data with all 8 methods | Task 3 Step 2 |
| Modal for create/edit (no metric_type field) | Task 3 Step 3 |
| Slug auto-fills from name on openNew | Task 3 Step 2 (onNameInput) |
| Slug disabled on edit | Task 3 Step 3 (`:disabled="editing !== null"`) |
| All .populations-* CSS classes | Task 3 Step 1 |
| formatCount helper (comma-formatted integer) | Task 3 Step 2 |
| formatAge helper | Task 3 Step 2 |
| Refresh button per-row with loading state | Task 3 Steps 2+3 |
| Empty state message | Task 3 Step 3 |
