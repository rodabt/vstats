# Tracker — Reuse / "Three Clicks and Done" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make experiment reuse the path of least resistance — clone opens the wizard pre-filled, any experiment can be saved as a reusable template, and new experiments can be started from a template in two clicks.

**Architecture:** Pure frontend change for clone (no API needed); new `experiment_templates` SQLite table + 3 REST endpoints for templates; "New from template" picker in the wizard entry point and "Save as template" in the sidebar.

**Tech Stack:** V / veb / SQLite (backend), Alpine.js (frontend), no new deps.

## Global Constraints

- All backend code goes in `apps/tracker/main.v` (single-file convention; ~1370 lines currently).
- V ORM uses `@[table: 'name']` struct attributes; migrations run in the `migrations` array inside `run_migrations()`.
- All routes use `@['/path'; method]` attribute above `pub fn (app &App) handler_name(mut ctx Context, ...) veb.Result`.
- Frontend: Alpine.js 3 store pattern. All store state lives in `Alpine.store('app', {...})` in `public/app.js`. HTML binds via `x-data`, `x-on`, `:class`, etc.
- `public/index.html` and `public/app.css` are served as static files; `public/app.js` requires a V recompile to take effect.
- Kill stale server before testing: `pkill -f apps/tracker && pkill -f "v run ." ; sleep 1 && v run . &`
- No new npm packages, no new V modules.

---

### Task 1: Clone Opens the Wizard Pre-Filled

Currently `cloneExp` in `app.js:400` immediately calls `saveExp`, silently creating a copy. This task changes it to open the modal pre-filled so the user can adjust name/dates before saving.

**Files:**
- Modify: `apps/tracker/public/app.js:400-425` (cloneExp function)

**Interfaces:**
- Consumes: `app.openModal(exp)` — already exists; opens `expModal` with `modalExp` set to the given experiment object.
- Produces: `cloneExp(exp)` — now calls `openModal` with a draft copy (id null, name prefixed, status reset).

- [ ] **Step 1: Update `cloneExp` in app.js**

Replace the existing `cloneExp` function (lines 400–425) with:

```js
cloneExp(exp) {
    // Open the wizard pre-filled; user adjusts name before saving.
    this.openModal({
        ...exp,
        id: null,
        name: `Copy of ${exp.name}`,
        status: 'planning',
        started_at: '',
        completed_at: '',
    });
},
```

- [ ] **Step 2: Verify manually**

Start the server:
```bash
pkill -f apps/tracker && pkill -f "v run ." ; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . &
sleep 3
```

Open `http://localhost:8080`. Click "Clone" on any card — the wizard should open pre-filled with "Copy of <name>" and status reset to planning. Saving should create a new experiment (not overwrite the original).

- [ ] **Step 3: Commit**

```bash
git add apps/tracker/public/app.js
git commit -m "feat(tracker): clone opens wizard pre-filled instead of silent copy"
```

---

### Task 2: Templates — Backend

Add an `experiment_templates` table and three endpoints: list, create-from-experiment, delete.

**Files:**
- Modify: `apps/tracker/main.v` — add struct, 3 migration lines, 3 route handlers, 1 request struct.

**Interfaces:**
- Produces:
  - `GET /api/templates` → `[]ExperimentTemplate` JSON
  - `POST /api/experiments/:id/save-as-template` → `ExperimentTemplate` JSON (201)
  - `DELETE /api/templates/:id` → plain text "deleted" (200)

- [ ] **Step 1: Add the `ExperimentTemplate` struct and request struct**

After the `Learning` struct (around line 90 in `main.v`), add:

```v
@[table: 'experiment_templates']
struct ExperimentTemplate {
	id                    int    @[primary; sql: serial]
	name                  string
	description           string
	test_type             string
	primary_metric_slug   string
	population_slugs      string
	guardrail_slugs       string
	learning_metric_slugs string
	variants              string
	statistical_test      string
	power                 f64
	significance_level    f64
	mde                   string
	duration_estimate     string
	flag_key              string
	event_lineage         string
	created_at            string
}
```

- [ ] **Step 2: Add migration**

In the `migrations` array inside `run_migrations()` (around line 969), add a new entry **before** the closing `]`:

```v
"CREATE TABLE IF NOT EXISTS experiment_templates (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, description TEXT DEFAULT '', test_type TEXT DEFAULT '', primary_metric_slug TEXT DEFAULT '', population_slugs TEXT DEFAULT '[]', guardrail_slugs TEXT DEFAULT '[]', learning_metric_slugs TEXT DEFAULT '[]', variants TEXT DEFAULT '[]', statistical_test TEXT DEFAULT '', power REAL DEFAULT 80, significance_level REAL DEFAULT 0.05, mde TEXT DEFAULT '', duration_estimate TEXT DEFAULT '', flag_key TEXT DEFAULT '', event_lineage TEXT DEFAULT '', created_at TEXT DEFAULT '')",
```

- [ ] **Step 3: Add the three route handlers**

Add these functions anywhere before `get_current_time()` (end of route section):

```v
@['/api/templates'; get]
pub fn (app &App) list_templates(mut ctx Context) veb.Result {
	templates := sql app.db {
		select from ExperimentTemplate
	} or {
		return ctx.server_error('failed to fetch templates')
	}
	return ctx.json(templates)
}

@['/api/experiments/:id/save-as-template'; post]
pub fn (app &App) save_as_template(mut ctx Context, id int) veb.Result {
	exps := sql app.db {
		select from Experiment where id == id
	} or {
		return ctx.server_error('db error')
	}
	if exps.len == 0 {
		return ctx.not_found()
	}
	e := exps[0]
	tmpl := ExperimentTemplate{
		name:                  e.name
		description:           e.description
		test_type:             e.test_type
		primary_metric_slug:   e.primary_metric_slug
		population_slugs:      e.population_slugs
		guardrail_slugs:       e.guardrail_slugs
		learning_metric_slugs: e.learning_metric_slugs
		variants:              e.variants
		statistical_test:      e.statistical_test
		power:                 e.power
		significance_level:    e.significance_level
		mde:                   e.mde
		duration_estimate:     e.duration_estimate
		flag_key:              e.flag_key
		event_lineage:         e.event_lineage
		created_at:            get_current_time()
	}
	sql app.db {
		insert tmpl into ExperimentTemplate
	} or {
		return ctx.server_error('failed to save template')
	}
	last_id := app.db.last_id()
	all := sql app.db {
		select from ExperimentTemplate
	} or { return ctx.server_error('fetch failed') }
	mut created := ExperimentTemplate{}
	for t in all {
		if t.id == last_id {
			created = t
			break
		}
	}
	ctx.res.set_status(.created)
	return ctx.json(created)
}

@['/api/templates/:id'; delete]
pub fn (app &App) delete_template(mut ctx Context, id int) veb.Result {
	sql app.db {
		delete from ExperimentTemplate where id == id
	} or {
		return ctx.server_error('db error')
	}
	return ctx.text('deleted')
}
```

- [ ] **Step 4: Compile and smoke-test the endpoints**

```bash
pkill -f apps/tracker && pkill -f "v run ." ; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . &
sleep 4
```

```bash
# List templates (empty at first)
curl -s http://localhost:8080/api/templates
# Expected: []

# Save experiment 1 as template
curl -s -X POST http://localhost:8080/api/experiments/1/save-as-template
# Expected: {"id":1,"name":"<name of exp 1>", ...}

# List templates (now has 1)
curl -s http://localhost:8080/api/templates | python3 -m json.tool | head -10
# Expected: array with 1 item

# Delete template 1
curl -s -X DELETE http://localhost:8080/api/templates/1
# Expected: deleted
```

- [ ] **Step 5: Commit**

```bash
git add apps/tracker/main.v
git commit -m "feat(tracker): add experiment_templates table and CRUD endpoints"
```

---

### Task 3: Templates — Frontend

Add template state to the Alpine store, a "Save as Template" button in the sidebar, and a "New from Template" entry point in the wizard.

**Files:**
- Modify: `apps/tracker/public/app.js` — add `templates` state, `loadTemplates`, `saveAsTemplate`, `newFromTemplate` methods.
- Modify: `apps/tracker/public/index.html` — add "Save as template" sidebar button; add template picker before the normal wizard header.

**Interfaces:**
- Consumes: `GET /api/templates`, `POST /api/experiments/:id/save-as-template`, `DELETE /api/templates/:id`
- Consumes: `app.openModal(templateObj)` — existing function; works with a template object if given an object without `id`.

- [ ] **Step 1: Add template state and methods to the Alpine store (`app.js`)**

In `Alpine.store('app', {...})`, find the `templates: []` line (already declared at line ~241 as `templates: []`). If it doesn't exist, add it alongside `metrics: []`. Then add these methods:

```js
templates: [],

async loadTemplates() {
    this.templates = await fetch('/api/templates').then(r => r.json()).catch(() => []);
},

async saveAsTemplate(exp) {
    if (!exp?.id) return;
    const resp = await fetch(`/api/experiments/${exp.id}/save-as-template`, { method: 'POST' });
    if (resp.ok) {
        await this.loadTemplates();
        this.toast('Saved as template');
    } else {
        this.toast('Failed to save template', 'error');
    }
},

async deleteTemplate(id) {
    if (!await this.confirm({ title: 'Delete template?', body: 'This cannot be undone.', confirmLabel: 'Delete', danger: true })) return;
    await fetch(`/api/templates/${id}`, { method: 'DELETE' });
    await this.loadTemplates();
    this.toast('Template deleted');
},

newFromTemplate(tmpl) {
    // Open the wizard seeded from template fields (no id → creates new experiment)
    this.openModal({
        id: null,
        name: tmpl.name,
        description: tmpl.description,
        test_type: tmpl.test_type,
        primary_metric_slug: tmpl.primary_metric_slug,
        population_slugs: tmpl.population_slugs,
        guardrail_slugs: tmpl.guardrail_slugs,
        learning_metric_slugs: tmpl.learning_metric_slugs,
        variants: tmpl.variants,
        statistical_test: tmpl.statistical_test,
        power: tmpl.power,
        significance_level: tmpl.significance_level,
        mde: tmpl.mde,
        duration_estimate: tmpl.duration_estimate,
        flag_key: tmpl.flag_key,
        event_lineage: tmpl.event_lineage,
        status: 'planning',
        started_at: '',
        completed_at: '',
    });
},
```

- [ ] **Step 2: Load templates on init**

In the `async init()` method of the store, add `this.loadTemplates()` alongside the existing `this.loadMetrics()` / `this.loadOwners()` calls:

```js
async init() {
    document.documentElement.setAttribute('data-theme', this.theme === 'light' ? 'light' : '');
    this.experiments = await fetch('/api/experiments').then(r => r.json()).catch(() => []);
    await Promise.all([
        this.loadMetrics(),
        this.loadPopulations(),
        this.loadOwners(),
        this.loadTemplates(),   // ← add this
    ]);
    // ... rest of init
},
```

- [ ] **Step 3: Add "Save as Template" to the sidebar action bar (`index.html`)**

Find the sidebar actions div (around line 438–451):

```html
<div class="sidebar-actions">
    <button class="sidebar-action-btn" @click="$store.app.openModal($store.app.selectedExp)">
        <span x-html="$store.app.Icons.Edit()"></span> Edit
    </button>
    <button class="sidebar-action-btn" @click="$store.app.cloneExp($store.app.selectedExp)">
        <span x-html="$store.app.Icons.Copy()"></span> Clone
    </button>
```

Add after the Clone button and before Archive:

```html
    <button class="sidebar-action-btn" @click="$store.app.saveAsTemplate($store.app.selectedExp)" title="Save as reusable template">
        <span x-html="$store.app.Icons.BarChart()"></span> Template
    </button>
```

- [ ] **Step 4: Add "New from template" picker to the wizard**

In `index.html`, find the modal header / first step of the wizard. The `expModal` Alpine data starts around where you see `x-data="expModal()"`. Find the tab content for `tab === 'General'` and add a template picker **above** the name field — it should only appear when `!form.id` (new experiment, not edit):

```html
<template x-if="!form.id && $store.app.templates.length > 0">
    <div class="ff" style="margin-bottom:18px;">
        <label class="ff-label">Start from template</label>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">
            <template x-for="tmpl in $store.app.templates" :key="tmpl.id">
                <button
                    type="button"
                    class="badge"
                    style="cursor:pointer;padding:4px 10px;background:var(--surface-sunken);border:1px solid var(--border);border-radius:6px;font-size:12px;font-weight:500;color:var(--text-secondary);"
                    @click="$store.app.newFromTemplate(tmpl)"
                    x-text="tmpl.name">
                </button>
            </template>
        </div>
    </div>
</template>
```

Place this immediately before the `<div class="ff">` that contains the experiment name input.

- [ ] **Step 5: Manual test**

```bash
pkill -f apps/tracker && pkill -f "v run ." ; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . &
sleep 4
```

1. Select any experiment → sidebar → click "Template" → toast "Saved as template".
2. Click "New Experiment" — the template picker chips appear above the name field.
3. Click a chip — the form fills with the template's values.
4. Change the name, click Save → new experiment created.
5. Verify the original experiment is untouched.

- [ ] **Step 6: Commit**

```bash
git add apps/tracker/public/app.js apps/tracker/public/index.html
git commit -m "feat(tracker): templates — save from sidebar, new from picker in wizard"
```
