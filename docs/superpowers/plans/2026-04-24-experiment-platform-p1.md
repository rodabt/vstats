# Experiment Platform — Phase 1: Foundation + Experiments Section

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **v-language skill:** Use the `v-language` skill for all V code questions, and read `v-language/veb-framework.md` and `v-language/database-orm.md` for patterns.

**Goal:** Add SQLite persistence and an Experiments section (Kanban + Timeline + detail view) to the existing vstats veb server, enabling full experiment lifecycle management.

**Architecture:** Extend `web/main.v` to open a SQLite DB on startup; add `web/db.v` for ORM models; add `web/api_experiments.v` for CRUD, results, and learnings endpoints; add new page routes and templates for `/experiments` and `/experiments/:id`. The existing App struct gains a `db sqlite.DB` field. All frontend interactions are Alpine.js calling the new REST API.

**Tech Stack:** V + veb, SQLite (db.sqlite ORM), Alpine.js, existing `web/static/css/style.css`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `web/db.v` | Create | ORM model structs + `setup_db()` |
| `web/main.v` | Modify | Open DB, call `setup_db`, add `db` field to App |
| `web/api_experiments.v` | Create | CRUD, results, learnings REST endpoints |
| `web/pages.v` | Modify | Add `/experiments`, `/experiments/:id`, `/learn`, `/dashboard` routes |
| `web/templates/_header.html` | Modify | Replace calculator nav with 4-section nav |
| `web/templates/experiments.html` | Create | Kanban + Timeline Alpine.js UI |
| `web/templates/experiment_detail.html` | Create | 3-tab detail view (Overview, Results, Learnings) |
| `web/static/js/experiments.js` | Create | Alpine component functions for experiments UI |

---

## Task 1: DB Models + App Struct

**Files:**
- Create: `web/db.v`
- Modify: `web/main.v`

### Step 1.1: Write `web/db.v` with ORM models

```v
module main

import db.sqlite
import time

@[table: 'experiments']
struct Experiment {
	id                    int    @[primary; sql: serial]
	name                  string
	team                  string
	product_area          string
	hypothesis            string
	treatment_description string
	control_description   string
	primary_metric        string
	metric_type           string
	guardrail_metrics     string
	test_type             string
	expected_effect_size  f64
	alpha                 f64
	stat_power            f64
	sample_size_per_group int
	status                string
	outcome               string
	notes                 string
	created_at            string
	updated_at            string
	started_at            string
	completed_at          string
}

@[table: 'experiment_results']
struct ExperimentResult {
	id              int    @[primary; sql: serial]
	experiment_id   int
	calculator_type string
	result_json     string
	significant     bool
	p_value         f64
	effect_size     f64
	ci_lower        f64
	ci_upper        f64
	created_at      string
}

@[table: 'learnings']
struct Learning {
	id            int    @[primary; sql: serial]
	experiment_id int
	learning_text string
	tags          string
	created_at    string
}

fn setup_db(mut db sqlite.DB) ! {
	sql db {
		create table Experiment
	}!
	sql db {
		create table ExperimentResult
	}!
	sql db {
		create table Learning
	}!
}

fn now_str() string {
	return time.now().format_ss()
}
```

### Step 1.2: Update `web/main.v`

Replace the current content:

```v
module main

import veb
import db.sqlite
import os

pub struct Context {
	veb.Context
}

pub struct App {
	veb.StaticHandler
pub mut:
	db sqlite.DB
}

fn main() {
	db_path := os.join_path(@VMODROOT, 'vstats.db')
	mut db := sqlite.connect(db_path) or { panic('DB connect failed: ${err}') }
	setup_db(mut db) or { panic('DB setup failed: ${err}') }

	mut app := &App{
		db: db
	}
	static_dir := os.join_path(@VMODROOT, 'web', 'static')
	app.mount_static_folder_at(static_dir, '/static') or { panic(err) }
	veb.run[App, Context](mut app, 8080)
}
```

### Step 1.3: Build to verify compilation

```bash
cd /home/rabt/devel/vstats && v build web/
```

Expected: exits 0, no errors.

### Step 1.4: Run server and verify DB file is created

```bash
cd /home/rabt/devel/vstats && v run web/ &
sleep 1
ls -la vstats.db
kill %1
```

Expected: `vstats.db` exists, non-zero size.

### Step 1.5: Commit

```bash
git add web/db.v web/main.v
git commit -m "feat(web): add SQLite DB models and setup for experiment platform"
```

---

## Task 2: Experiments CRUD API

**Files:**
- Create: `web/api_experiments.v`
- Modify: `web/pages.v` (routes added in Task 4)

### Step 2.1: Create `web/api_experiments.v`

```v
module main

import veb
import json

// --- Request / Response structs ---

struct CreateExperimentRequest {
pub mut:
	name                  string
	team                  string
	product_area          string
	hypothesis            string
	treatment_description string
	control_description   string
	primary_metric        string
	metric_type           string
	guardrail_metrics     string
	test_type             string
	expected_effect_size  f64
	alpha                 f64 = 0.05
	stat_power            f64 = 0.80
	sample_size_per_group int
	notes                 string
}

struct UpdateExperimentRequest {
pub mut:
	name                  string
	team                  string
	product_area          string
	hypothesis            string
	treatment_description string
	control_description   string
	primary_metric        string
	metric_type           string
	guardrail_metrics     string
	test_type             string
	expected_effect_size  f64
	alpha                 f64
	stat_power            f64
	sample_size_per_group int
	notes                 string
}

struct UpdateStatusRequest {
pub mut:
	status  string
	outcome string
}

struct SaveResultRequest {
pub mut:
	calculator_type string
	result_json     string
	significant     bool
	p_value         f64
	effect_size     f64
	ci_lower        f64
	ci_upper        f64
}

struct AddLearningRequest {
pub mut:
	learning_text string
	tags          string
}

// --- List experiments ---

@['/api/experiments'; get]
pub fn (app &App) api_list_experiments(mut ctx Context) veb.Result {
	experiments := sql app.db {
		select from Experiment
	} or {
		return ctx.server_error('db error')
	}
	return ctx.json(experiments)
}

// --- Get one experiment ---

@['/api/experiments/:id'; get]
pub fn (app &App) api_get_experiment(mut ctx Context, id int) veb.Result {
	rows := sql app.db {
		select from Experiment where id == id
	} or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.not_found()
	}
	return ctx.json(rows[0])
}

// --- Create experiment ---

@['/api/experiments'; post]
pub fn (app &App) api_create_experiment(mut ctx Context) veb.Result {
	req := json.decode(CreateExperimentRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.name == '' {
		return api_error(mut ctx, 'name is required')
	}
	ts := now_str()
	exp := Experiment{
		name:                  req.name
		team:                  req.team
		product_area:          req.product_area
		hypothesis:            req.hypothesis
		treatment_description: req.treatment_description
		control_description:   req.control_description
		primary_metric:        req.primary_metric
		metric_type:           req.metric_type
		guardrail_metrics:     req.guardrail_metrics
		test_type:             req.test_type
		expected_effect_size:  req.expected_effect_size
		alpha:                 if req.alpha > 0 { req.alpha } else { 0.05 }
		stat_power:            if req.stat_power > 0 { req.stat_power } else { 0.80 }
		sample_size_per_group: req.sample_size_per_group
		notes:                 req.notes
		status:                'plan'
		created_at:            ts
		updated_at:            ts
	}
	sql app.db {
		insert exp into Experiment
	} or {
		return ctx.server_error('insert failed')
	}
	last := sql app.db {
		select from Experiment order by id desc limit 1
	} or {
		return ctx.server_error('db error')
	}
	ctx.res.set_status(.created)
	return ctx.json(last[0])
}

// --- Update experiment fields ---

@['/api/experiments/:id'; put]
pub fn (app &App) api_update_experiment(mut ctx Context, id int) veb.Result {
	rows := sql app.db {
		select from Experiment where id == id
	} or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.not_found()
	}
	req := json.decode(UpdateExperimentRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	ts := now_str()
	sql app.db {
		update Experiment set name = req.name, team = req.team,
			product_area = req.product_area, hypothesis = req.hypothesis,
			treatment_description = req.treatment_description,
			control_description = req.control_description,
			primary_metric = req.primary_metric, metric_type = req.metric_type,
			guardrail_metrics = req.guardrail_metrics, test_type = req.test_type,
			expected_effect_size = req.expected_effect_size, alpha = req.alpha,
			stat_power = req.stat_power,
			sample_size_per_group = req.sample_size_per_group,
			notes = req.notes, updated_at = ts
		where id == id
	} or {
		return ctx.server_error('update failed')
	}
	updated := sql app.db {
		select from Experiment where id == id
	} or {
		return ctx.server_error('db error')
	}
	return ctx.json(updated[0])
}

// --- Update status ---

@['/api/experiments/:id/status'; put]
pub fn (app &App) api_update_status(mut ctx Context, id int) veb.Result {
	rows := sql app.db {
		select from Experiment where id == id
	} or {
		return ctx.server_error('db error')
	}
	if rows.len == 0 {
		return ctx.not_found()
	}
	req := json.decode(UpdateStatusRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	valid_statuses := ['plan', 'instrumentation', 'running', 'completed', 'paused', 'cancelled']
	if req.status !in valid_statuses {
		return api_error(mut ctx, 'invalid status')
	}
	ts := now_str()
	mut started := rows[0].started_at
	mut completed := rows[0].completed_at
	if req.status == 'running' && started == '' {
		started = ts
	}
	if req.status in ['completed', 'paused', 'cancelled'] {
		completed = ts
	}
	sql app.db {
		update Experiment set status = req.status, outcome = req.outcome,
			started_at = started, completed_at = completed, updated_at = ts
		where id == id
	} or {
		return ctx.server_error('update failed')
	}
	updated := sql app.db {
		select from Experiment where id == id
	} or {
		return ctx.server_error('db error')
	}
	return ctx.json(updated[0])
}

// --- Delete experiment ---

@['/api/experiments/:id'; delete]
pub fn (app &App) api_delete_experiment(mut ctx Context, id int) veb.Result {
	sql app.db {
		delete from Experiment where id == id
	} or {
		return ctx.server_error('delete failed')
	}
	sql app.db {
		delete from ExperimentResult where experiment_id == id
	} or {}
	sql app.db {
		delete from Learning where experiment_id == id
	} or {}
	return ctx.json(map[string]string{})
}

// --- Results ---

@['/api/experiments/:id/results'; get]
pub fn (app &App) api_list_results(mut ctx Context, id int) veb.Result {
	results := sql app.db {
		select from ExperimentResult where experiment_id == id
	} or {
		return ctx.server_error('db error')
	}
	return ctx.json(results)
}

@['/api/experiments/:id/results'; post]
pub fn (app &App) api_save_result(mut ctx Context, id int) veb.Result {
	req := json.decode(SaveResultRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.calculator_type == '' {
		return api_error(mut ctx, 'calculator_type is required')
	}
	result := ExperimentResult{
		experiment_id:   id
		calculator_type: req.calculator_type
		result_json:     req.result_json
		significant:     req.significant
		p_value:         req.p_value
		effect_size:     req.effect_size
		ci_lower:        req.ci_lower
		ci_upper:        req.ci_upper
		created_at:      now_str()
	}
	sql app.db {
		insert result into ExperimentResult
	} or {
		return ctx.server_error('insert failed')
	}
	ctx.res.set_status(.created)
	return ctx.json(result)
}

// --- Learnings ---

@['/api/experiments/:id/learnings'; get]
pub fn (app &App) api_list_learnings(mut ctx Context, id int) veb.Result {
	learnings := sql app.db {
		select from Learning where experiment_id == id
	} or {
		return ctx.server_error('db error')
	}
	return ctx.json(learnings)
}

@['/api/experiments/:id/learnings'; post]
pub fn (app &App) api_add_learning(mut ctx Context, id int) veb.Result {
	req := json.decode(AddLearningRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.learning_text == '' {
		return api_error(mut ctx, 'learning_text is required')
	}
	learning := Learning{
		experiment_id: id
		learning_text: req.learning_text
		tags:          req.tags
		created_at:    now_str()
	}
	sql app.db {
		insert learning into Learning
	} or {
		return ctx.server_error('insert failed')
	}
	ctx.res.set_status(.created)
	return ctx.json(learning)
}
```

### Step 2.2: Build to verify

```bash
cd /home/rabt/devel/vstats && v build web/
```

Expected: exits 0.

### Step 2.3: Smoke-test the API

Start the server in background, then run:

```bash
cd /home/rabt/devel/vstats && v run web/ &
sleep 1

# Create an experiment
curl -s -X POST http://localhost:8080/api/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"Test Onboarding CTA","team":"Growth","metric_type":"binary","primary_metric":"signup_rate","test_type":"ab_test","alpha":0.05,"stat_power":0.8}' | python3 -m json.tool

# List experiments
curl -s http://localhost:8080/api/experiments | python3 -m json.tool

# Update status
curl -s -X PUT http://localhost:8080/api/experiments/1/status \
  -H 'Content-Type: application/json' \
  -d '{"status":"running"}' | python3 -m json.tool

kill %1
```

Expected: experiment created with `id:1`, `status:"plan"`; status update returns `status:"running"` with `started_at` set.

### Step 2.4: Commit

```bash
git add web/api_experiments.v
git commit -m "feat(web): add experiments CRUD, results, and learnings REST API"
```

---

## Task 3: Page Routes

**Files:**
- Modify: `web/pages.v`

### Step 3.1: Add new routes to `web/pages.v`

Add to the bottom of `web/pages.v` (after the existing `deff_page` route):

```v
@['/experiments'; get]
pub fn (app &App) experiments_page(mut ctx Context) veb.Result {
	html := read_template('experiments') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/experiments/:id'; get]
pub fn (app &App) experiment_detail_page(mut ctx Context, id int) veb.Result {
	html := read_template('experiment_detail') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/learn'; get]
pub fn (app &App) learn_page(mut ctx Context) veb.Result {
	html := read_template('learn') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/dashboard'; get]
pub fn (app &App) dashboard_page(mut ctx Context) veb.Result {
	html := read_template('dashboard') or { return ctx.server_error('template error') }
	return ctx.html(html)
}
```

Also update the index route to redirect to `/experiments`:

```v
@['/'; get]
pub fn (app &App) index(mut ctx Context) veb.Result {
	return ctx.redirect('/experiments')
}
```

### Step 3.2: Build to verify

```bash
cd /home/rabt/devel/vstats && v build web/
```

Expected: exits 0 (templates don't exist yet, but routes compile fine).

### Step 3.3: Commit

```bash
git add web/pages.v
git commit -m "feat(web): add experiment platform page routes"
```

---

## Task 4: Navigation Update

**Files:**
- Modify: `web/templates/_header.html`

### Step 4.1: Update the nav

Replace the nav block (lines 13-24 in `_header.html`):

```html
<nav>
  <a class="logo" href="/experiments"><span>v</span>stats</a>
  <a href="/experiments">Experiments</a>
  <a href="/calculators/ab-test">Calculators</a>
  <a href="/learn">Learn</a>
  <a href="/dashboard">Dashboard</a>
</nav>
```

The "Calculators" link points to the existing AB test page as the calculator entry point (the calculator index page can be updated in a follow-on task).

### Step 4.2: Build and visual check

```bash
cd /home/rabt/devel/vstats && v run web/ &
sleep 1
# Open http://localhost:8080/calculators/ab-test in browser and verify nav shows 4 items
kill %1
```

### Step 4.3: Commit

```bash
git add web/templates/_header.html
git commit -m "feat(web): update nav to 4-section experiment platform layout"
```

---

## Task 4b: Stub Templates for Learn + Dashboard

**Files:**
- Create: `web/templates/learn.html`
- Create: `web/templates/dashboard.html`

### Step 4b.1: Write stub templates

`web/templates/learn.html`:
```html
<div class="page-header"><h1>Learn</h1></div>
<p>Coming in Phase 3.</p>
```

`web/templates/dashboard.html`:
```html
<div class="page-header"><h1>Dashboard</h1></div>
<p>Coming in Phase 3.</p>
```

### Step 4b.2: Commit

```bash
git add web/templates/learn.html web/templates/dashboard.html
git commit -m "chore(web): add stub Learn and Dashboard templates"
```

---

## Task 5: Experiments Alpine Component (JS)

**Files:**
- Create: `web/static/js/experiments.js`

### Step 5.1: Write `web/static/js/experiments.js`

```javascript
function experimentsApp() {
  return {
    view: 'kanban',
    experiments: [],
    loading: false,
    showCreateModal: false,
    groupBy: 'team',

    columns: [
      { label: 'Plan', statuses: ['plan'] },
      { label: 'Instrumentation', statuses: ['instrumentation'] },
      { label: 'Running', statuses: ['running'] },
      { label: 'Completed', statuses: ['completed', 'paused', 'cancelled'] },
    ],

    statusColors: {
      plan: 'badge-blue',
      instrumentation: 'badge-yellow',
      running: 'badge-green',
      completed: 'badge-gray',
      paused: 'badge-orange',
      cancelled: 'badge-red',
    },

    createForm: {
      name: '', team: '', product_area: '', hypothesis: '',
      treatment_description: '', control_description: '',
      primary_metric: '', metric_type: 'binary', test_type: 'ab_test',
      expected_effect_size: 0, alpha: 0.05, stat_power: 0.80,
      sample_size_per_group: 0, notes: '',
    },

    async init() {
      await this.loadExperiments();
    },

    async loadExperiments() {
      this.loading = true;
      try {
        const res = await fetch('/api/experiments');
        this.experiments = await res.json();
      } finally {
        this.loading = false;
      }
    },

    byStatus(statuses) {
      return this.experiments.filter(e => statuses.includes(e.status));
    },

    daysInStage(exp) {
      const ref = exp.started_at || exp.created_at;
      if (!ref) return 0;
      const start = new Date(ref.replace(' ', 'T'));
      const end = exp.completed_at ? new Date(exp.completed_at.replace(' ', 'T')) : new Date();
      return Math.floor((end - start) / 86400000);
    },

    openDetail(id) {
      window.location.href = `/experiments/${id}`;
    },

    openCreate() {
      this.createForm = {
        name: '', team: '', product_area: '', hypothesis: '',
        treatment_description: '', control_description: '',
        primary_metric: '', metric_type: 'binary', test_type: 'ab_test',
        expected_effect_size: 0, alpha: 0.05, stat_power: 0.80,
        sample_size_per_group: 0, notes: '',
      };
      this.showCreateModal = true;
    },

    async submitCreate() {
      if (!this.createForm.name.trim()) return;
      const res = await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.createForm),
      });
      if (res.ok) {
        this.showCreateModal = false;
        await this.loadExperiments();
      }
    },

    // Timeline helpers
    timelineStart() {
      const dates = this.experiments
        .map(e => e.created_at)
        .filter(Boolean)
        .map(d => new Date(d.replace(' ', 'T')));
      if (dates.length === 0) return new Date();
      return new Date(Math.min(...dates));
    },

    timelineEnd() {
      const dates = this.experiments
        .map(e => e.completed_at || new Date().toISOString())
        .filter(Boolean)
        .map(d => new Date(d.replace(' ', 'T') || d));
      if (dates.length === 0) return new Date();
      return new Date(Math.max(...dates));
    },

    timelineBarStyle(exp) {
      const start = this.timelineStart();
      const end = this.timelineEnd();
      const totalMs = end - start || 1;
      const expStart = new Date((exp.created_at || '').replace(' ', 'T'));
      const expEnd = exp.completed_at
        ? new Date(exp.completed_at.replace(' ', 'T'))
        : new Date();
      const left = ((expStart - start) / totalMs) * 100;
      const width = Math.max(((expEnd - expStart) / totalMs) * 100, 1);
      return `left: ${left.toFixed(1)}%; width: ${width.toFixed(1)}%;`;
    },

    timelineGroups() {
      const key = this.groupBy;
      const groups = {};
      for (const exp of this.experiments) {
        const g = exp[key] || 'Unassigned';
        if (!groups[g]) groups[g] = [];
        groups[g].push(exp);
      }
      return Object.entries(groups).map(([label, items]) => ({ label, items }));
    },
  };
}
```

### Step 5.2: Commit

```bash
git add web/static/js/experiments.js
git commit -m "feat(web): add experiments Alpine.js component"
```

---

## Task 6: Experiments Page Template

**Files:**
- Create: `web/templates/experiments.html`

### Step 6.0: Extract the "New Experiment" modal from `experiment-tracker.html`

Before writing `experiments.html`, open `experiment-tracker.html` in a browser (or search for `new experiment` / `createExperiment` in the source) to locate the existing modal markup and its corresponding JS form handler. The exact field names and validation logic used there should be the source of truth for the create form in Step 6.1 — adapt to Alpine.js style rather than rewriting from scratch.

### Step 6.1: Write `web/templates/experiments.html`

```html
<script src="/static/js/experiments.js"></script>

<div x-data="experimentsApp()" class="experiments-page">

  <!-- Header bar -->
  <div class="page-header">
    <h1>Experiments</h1>
    <div class="header-actions">
      <div class="view-toggle">
        <button :class="view === 'kanban' ? 'active' : ''" @click="view = 'kanban'">Kanban</button>
        <button :class="view === 'timeline' ? 'active' : ''" @click="view = 'timeline'">Timeline</button>
      </div>
      <button class="btn-primary" @click="openCreate()">+ New Experiment</button>
    </div>
  </div>

  <div x-show="loading" class="loading">Loading experiments…</div>

  <!-- Kanban board -->
  <div x-show="!loading && view === 'kanban'" class="kanban-board">
    <template x-for="col in columns" :key="col.label">
      <div class="kanban-col">
        <div class="kanban-col-header">
          <span x-text="col.label"></span>
          <span class="count" x-text="byStatus(col.statuses).length"></span>
        </div>
        <div class="kanban-cards">
          <template x-for="exp in byStatus(col.statuses)" :key="exp.id">
            <div class="kanban-card" @click="openDetail(exp.id)">
              <div class="card-name" x-text="exp.name"></div>
              <div class="card-meta">
                <span x-text="exp.team"></span>
                <span x-show="exp.primary_metric" x-text="'· ' + exp.primary_metric"></span>
              </div>
              <div class="card-footer">
                <span :class="'badge ' + (statusColors[exp.status] || 'badge-gray')" x-text="exp.status"></span>
                <span class="days" x-text="daysInStage(exp) + 'd'"></span>
              </div>
            </div>
          </template>
          <div x-show="byStatus(col.statuses).length === 0" class="kanban-empty">—</div>
        </div>
      </div>
    </template>
  </div>

  <!-- Timeline view -->
  <div x-show="!loading && view === 'timeline'" class="timeline-view">
    <div class="timeline-controls">
      <label>Group by:
        <select x-model="groupBy">
          <option value="team">Team</option>
          <option value="product_area">Product Area</option>
        </select>
      </label>
    </div>
    <div class="timeline-chart">
      <template x-for="group in timelineGroups()" :key="group.label">
        <div class="timeline-group">
          <div class="timeline-group-label" x-text="group.label"></div>
          <div class="timeline-rows">
            <template x-for="exp in group.items" :key="exp.id">
              <div class="timeline-row">
                <div class="timeline-row-label" x-text="exp.name" @click="openDetail(exp.id)" style="cursor:pointer"></div>
                <div class="timeline-row-bar">
                  <div
                    class="timeline-bar"
                    :class="'status-' + exp.status"
                    :style="timelineBarStyle(exp)"
                    :title="exp.name + ' · ' + exp.status"
                    @click="openDetail(exp.id)"
                  ></div>
                </div>
              </div>
            </template>
          </div>
        </div>
      </template>
      <div x-show="experiments.length === 0" class="empty-state">No experiments yet. Create one to get started.</div>
    </div>
  </div>

  <!-- Create experiment modal -->
  <div x-show="showCreateModal" class="modal-overlay" @click.self="showCreateModal = false">
    <div class="modal">
      <div class="modal-header">
        <h2>New Experiment</h2>
        <button class="close-btn" @click="showCreateModal = false">×</button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label>Name *</label>
          <input x-model="createForm.name" type="text" placeholder="e.g. New onboarding CTA">
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Team</label>
            <input x-model="createForm.team" type="text" placeholder="e.g. Growth">
          </div>
          <div class="form-group">
            <label>Product Area</label>
            <input x-model="createForm.product_area" type="text" placeholder="e.g. Onboarding">
          </div>
        </div>
        <div class="form-group">
          <label>Hypothesis</label>
          <textarea x-model="createForm.hypothesis" rows="3" placeholder="We believe [change] will cause [metric] to [direction] because [reason]"></textarea>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Primary Metric</label>
            <input x-model="createForm.primary_metric" type="text" placeholder="e.g. signup_rate">
          </div>
          <div class="form-group">
            <label>Metric Type</label>
            <select x-model="createForm.metric_type">
              <option value="binary">Binary (proportion)</option>
              <option value="continuous">Continuous</option>
            </select>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Test Type</label>
            <select x-model="createForm.test_type">
              <option value="ab_test">A/B Test</option>
              <option value="cuped">CUPED</option>
              <option value="sprt">SPRT</option>
              <option value="psm">PSM</option>
              <option value="did">DiD</option>
              <option value="cluster">Cluster</option>
            </select>
          </div>
          <div class="form-group">
            <label>Alpha</label>
            <input x-model.number="createForm.alpha" type="number" step="0.01" min="0.01" max="0.2">
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" @click="showCreateModal = false">Cancel</button>
        <button class="btn-primary" @click="submitCreate()">Create Experiment</button>
      </div>
    </div>
  </div>

</div>
```

### Step 6.2: Build and visual check

```bash
cd /home/rabt/devel/vstats && v run web/ &
sleep 1
# Open http://localhost:8080/experiments in browser
# Verify: Kanban renders 4 columns, "New Experiment" button opens modal
# Create an experiment, verify it appears in the Plan column
kill %1
```

### Step 6.3: Commit

```bash
git add web/templates/experiments.html
git commit -m "feat(web): add experiments Kanban and Timeline view template"
```

---

## Task 7: Experiment Detail Template

**Files:**
- Create: `web/templates/experiment_detail.html`

### Step 7.1: Add Alpine component to `web/static/js/experiments.js`

Append to `web/static/js/experiments.js`:

```javascript
function experimentDetailApp() {
  return {
    exp: null,
    results: [],
    learnings: [],
    activeTab: 'overview',
    loading: true,
    editMode: false,
    showStatusModal: false,
    showLearningModal: false,
    statusForm: { status: '', outcome: '' },
    learningForm: { learning_text: '', tags: '' },
    editForm: {},

    statusTransitions: {
      plan: ['instrumentation', 'cancelled'],
      instrumentation: ['running', 'cancelled'],
      running: ['completed', 'paused', 'cancelled'],
      paused: ['running', 'cancelled'],
      completed: [],
      cancelled: [],
    },

    async init() {
      const id = window.location.pathname.split('/').pop();
      await Promise.all([
        this.loadExperiment(id),
        this.loadResults(id),
        this.loadLearnings(id),
      ]);
      this.loading = false;
    },

    async loadExperiment(id) {
      const res = await fetch(`/api/experiments/${id}`);
      if (res.ok) this.exp = await res.json();
    },

    async loadResults(id) {
      const res = await fetch(`/api/experiments/${id}/results`);
      if (res.ok) this.results = await res.json();
    },

    async loadLearnings(id) {
      const res = await fetch(`/api/experiments/${id}/learnings`);
      if (res.ok) this.learnings = await res.json();
    },

    startEdit() {
      this.editForm = { ...this.exp };
      this.editMode = true;
    },

    async saveEdit() {
      const res = await fetch(`/api/experiments/${this.exp.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.editForm),
      });
      if (res.ok) {
        this.exp = await res.json();
        this.editMode = false;
      }
    },

    openStatusChange(newStatus) {
      this.statusForm = { status: newStatus, outcome: '' };
      this.showStatusModal = true;
    },

    async submitStatusChange() {
      const res = await fetch(`/api/experiments/${this.exp.id}/status`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.statusForm),
      });
      if (res.ok) {
        this.exp = await res.json();
        this.showStatusModal = false;
      }
    },

    async addLearning() {
      if (!this.learningForm.learning_text.trim()) return;
      const res = await fetch(`/api/experiments/${this.exp.id}/learnings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.learningForm),
      });
      if (res.ok) {
        await this.loadLearnings(this.exp.id);
        this.learningForm = { learning_text: '', tags: '' };
        this.showLearningModal = false;
      }
    },

    nextStatuses() {
      if (!this.exp) return [];
      return this.statusTransitions[this.exp.status] || [];
    },

    fmt(v) {
      if (v === null || v === undefined || v === '') return '—';
      return typeof v === 'number' ? v.toFixed(4) : v;
    },
  };
}
```

### Step 7.2: Create `web/templates/experiment_detail.html`

```html
<script src="/static/js/experiments.js"></script>

<div x-data="experimentDetailApp()" class="detail-page">

  <div x-show="loading" class="loading">Loading…</div>

  <template x-if="!loading && exp">
    <div>
      <!-- Breadcrumb -->
      <div class="breadcrumb">
        <a href="/experiments">← Experiments</a>
      </div>

      <!-- Experiment header -->
      <div class="detail-header">
        <div>
          <h1 x-text="exp.name"></h1>
          <div class="detail-meta">
            <span x-text="exp.team"></span>
            <span x-show="exp.product_area"> · <span x-text="exp.product_area"></span></span>
          </div>
        </div>
        <div class="detail-actions">
          <span class="badge" :class="'badge-' + exp.status" x-text="exp.status"></span>
          <template x-for="s in nextStatuses()" :key="s">
            <button class="btn-secondary" @click="openStatusChange(s)">
              Move to <span x-text="s"></span>
            </button>
          </template>
          <button class="btn-secondary" @click="startEdit()" x-show="!editMode">Edit</button>
        </div>
      </div>

      <!-- Tabs -->
      <div class="tabs">
        <button :class="activeTab === 'overview' ? 'active' : ''" @click="activeTab = 'overview'">Overview</button>
        <button :class="activeTab === 'results' ? 'active' : ''" @click="activeTab = 'results'">
          Results <span class="count" x-text="results.length"></span>
        </button>
        <button :class="activeTab === 'learnings' ? 'active' : ''" @click="activeTab = 'learnings'">
          Learnings <span class="count" x-text="learnings.length"></span>
        </button>
      </div>

      <!-- Overview tab -->
      <div x-show="activeTab === 'overview'" class="tab-content">
        <template x-if="!editMode">
          <div class="overview-grid">
            <div class="field-group">
              <label>Hypothesis</label>
              <p x-text="exp.hypothesis || '—'"></p>
            </div>
            <div class="field-row">
              <div class="field-group">
                <label>Primary Metric</label>
                <p x-text="exp.primary_metric || '—'"></p>
              </div>
              <div class="field-group">
                <label>Metric Type</label>
                <p x-text="exp.metric_type || '—'"></p>
              </div>
              <div class="field-group">
                <label>Test Type</label>
                <p x-text="exp.test_type || '—'"></p>
              </div>
            </div>
            <div class="field-row">
              <div class="field-group">
                <label>MDE</label>
                <p x-text="fmt(exp.expected_effect_size)"></p>
              </div>
              <div class="field-group">
                <label>Alpha</label>
                <p x-text="fmt(exp.alpha)"></p>
              </div>
              <div class="field-group">
                <label>Power</label>
                <p x-text="fmt(exp.stat_power)"></p>
              </div>
              <div class="field-group">
                <label>Sample Size / Group</label>
                <p x-text="exp.sample_size_per_group || '—'"></p>
              </div>
            </div>
            <div class="field-group">
              <label>Treatment</label>
              <p x-text="exp.treatment_description || '—'"></p>
            </div>
            <div class="field-group">
              <label>Control</label>
              <p x-text="exp.control_description || '—'"></p>
            </div>
            <div class="field-group" x-show="exp.notes">
              <label>Notes</label>
              <p x-text="exp.notes"></p>
            </div>
          </div>
        </template>

        <template x-if="editMode">
          <div class="edit-form">
            <div class="form-group">
              <label>Name</label>
              <input x-model="editForm.name" type="text">
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>Team</label>
                <input x-model="editForm.team" type="text">
              </div>
              <div class="form-group">
                <label>Product Area</label>
                <input x-model="editForm.product_area" type="text">
              </div>
            </div>
            <div class="form-group">
              <label>Hypothesis</label>
              <textarea x-model="editForm.hypothesis" rows="3"></textarea>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>Primary Metric</label>
                <input x-model="editForm.primary_metric" type="text">
              </div>
              <div class="form-group">
                <label>Metric Type</label>
                <select x-model="editForm.metric_type">
                  <option value="binary">Binary</option>
                  <option value="continuous">Continuous</option>
                </select>
              </div>
              <div class="form-group">
                <label>Test Type</label>
                <select x-model="editForm.test_type">
                  <option value="ab_test">A/B Test</option>
                  <option value="cuped">CUPED</option>
                  <option value="sprt">SPRT</option>
                  <option value="psm">PSM</option>
                  <option value="did">DiD</option>
                  <option value="cluster">Cluster</option>
                </select>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>MDE</label>
                <input x-model.number="editForm.expected_effect_size" type="number" step="0.001">
              </div>
              <div class="form-group">
                <label>Alpha</label>
                <input x-model.number="editForm.alpha" type="number" step="0.01">
              </div>
              <div class="form-group">
                <label>Power</label>
                <input x-model.number="editForm.stat_power" type="number" step="0.01">
              </div>
              <div class="form-group">
                <label>Sample Size / Group</label>
                <input x-model.number="editForm.sample_size_per_group" type="number">
              </div>
            </div>
            <div class="form-group">
              <label>Treatment Description</label>
              <textarea x-model="editForm.treatment_description" rows="2"></textarea>
            </div>
            <div class="form-group">
              <label>Control Description</label>
              <textarea x-model="editForm.control_description" rows="2"></textarea>
            </div>
            <div class="form-group">
              <label>Notes</label>
              <textarea x-model="editForm.notes" rows="3"></textarea>
            </div>
            <div class="form-actions">
              <button class="btn-secondary" @click="editMode = false">Cancel</button>
              <button class="btn-primary" @click="saveEdit()">Save</button>
            </div>
          </div>
        </template>
      </div>

      <!-- Results tab -->
      <div x-show="activeTab === 'results'" class="tab-content">
        <div x-show="results.length === 0" class="empty-state">
          No results saved yet.
          <a :href="'/calculators/ab-test?exp=' + exp.id">Run a calculator →</a>
        </div>
        <template x-for="r in results" :key="r.id">
          <div class="result-card">
            <div class="result-header">
              <span class="calculator-type" x-text="r.calculator_type.replace('_', ' ')"></span>
              <span :class="r.significant ? 'badge-green badge' : 'badge-gray badge'"
                    x-text="r.significant ? 'Significant' : 'Not significant'"></span>
              <span class="result-date" x-text="r.created_at"></span>
            </div>
            <div class="result-stats">
              <div class="stat"><label>p-value</label><span x-text="fmt(r.p_value)"></span></div>
              <div class="stat"><label>Effect size</label><span x-text="fmt(r.effect_size)"></span></div>
              <div class="stat"><label>CI lower</label><span x-text="fmt(r.ci_lower)"></span></div>
              <div class="stat"><label>CI upper</label><span x-text="fmt(r.ci_upper)"></span></div>
            </div>
          </div>
        </template>
      </div>

      <!-- Learnings tab -->
      <div x-show="activeTab === 'learnings'" class="tab-content">
        <div class="tab-actions">
          <button class="btn-primary" @click="showLearningModal = true">+ Add Learning</button>
        </div>
        <div x-show="learnings.length === 0" class="empty-state">No learnings recorded yet.</div>
        <template x-for="l in learnings" :key="l.id">
          <div class="learning-card">
            <p x-text="l.learning_text"></p>
            <div class="learning-meta" x-show="l.tags">
              <template x-for="tag in (l.tags ? JSON.parse(l.tags) : [])" :key="tag">
                <span class="tag" x-text="tag"></span>
              </template>
            </div>
            <div class="learning-date" x-text="l.created_at"></div>
          </div>
        </template>
      </div>
    </div>
  </template>

  <!-- Status change modal -->
  <div x-show="showStatusModal" class="modal-overlay" @click.self="showStatusModal = false">
    <div class="modal">
      <div class="modal-header">
        <h2>Move to <span x-text="statusForm.status"></span></h2>
      </div>
      <div class="modal-body">
        <template x-if="statusForm.status === 'completed'">
          <div>
            <div class="form-group">
              <label>Outcome *</label>
              <select x-model="statusForm.outcome">
                <option value="">Select outcome…</option>
                <option value="significant_positive">Significant — positive</option>
                <option value="significant_negative">Significant — negative</option>
                <option value="null">Null (no significant effect)</option>
              </select>
            </div>
            <p class="hint">You'll be able to add a learning on the next screen.</p>
          </div>
        </template>
        <template x-if="statusForm.status !== 'completed'">
          <p>Confirm moving this experiment to <strong x-text="statusForm.status"></strong>?</p>
        </template>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" @click="showStatusModal = false">Cancel</button>
        <button class="btn-primary"
          :disabled="statusForm.status === 'completed' && !statusForm.outcome"
          @click="submitStatusChange()">Confirm</button>
      </div>
    </div>
  </div>

  <!-- Add learning modal -->
  <div x-show="showLearningModal" class="modal-overlay" @click.self="showLearningModal = false">
    <div class="modal">
      <div class="modal-header">
        <h2>Add Learning</h2>
        <button class="close-btn" @click="showLearningModal = false">×</button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label>What did we learn? *</label>
          <textarea x-model="learningForm.learning_text" rows="4"
            placeholder="e.g. Simplifying the CTA copy increased signups by 12% among mobile users but had no effect on desktop."></textarea>
        </div>
        <div class="form-group">
          <label>Tags (comma-separated)</label>
          <input x-model="learningForm.tags" type="text" placeholder="e.g. mobile, cta, onboarding">
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" @click="showLearningModal = false">Cancel</button>
        <button class="btn-primary" @click="addLearning()">Save Learning</button>
      </div>
    </div>
  </div>

</div>
```

> **Note on tags:** The `learningForm.tags` field captures a comma-separated string from the user. The backend stores it as-is. When adding learning via the API, the frontend should pre-process to JSON if the API expects JSON array — or keep as a plain string and parse at display time. The template above parses it as JSON; therefore the `api_add_learning` handler should accept it as a plain string and store as-is. To ensure valid JSON storage, update `api_add_learning` to convert comma-separated string to JSON array before storing:

In `web/api_experiments.v`, update the `api_add_learning` handler to convert tags:

```v
// Convert comma-separated tags to JSON array before storing
tags_json := if req.tags.trim_space() == '' {
    '[]'
} else {
    '[' + req.tags.split(',').map('"' + it.trim_space() + '"').join(',') + ']'
}
learning := Learning{
    experiment_id: id
    learning_text: req.learning_text
    tags:          tags_json
    created_at:    now_str()
}
```

### Step 7.3: Build and end-to-end manual test

```bash
cd /home/rabt/devel/vstats && v run web/ &
sleep 1
# Open http://localhost:8080/experiments
# Create experiment "Homepage Hero Test" (team: Growth, metric: conversion_rate, binary)
# Click card → opens /experiments/1
# Verify: tabs render, Overview shows fields
# Click "Move to instrumentation" → confirm
# Click Learnings tab → "Add Learning" → add one → verify it appears
kill %1
```

### Step 7.4: Commit

```bash
git add web/templates/experiment_detail.html web/static/js/experiments.js
git commit -m "feat(web): add experiment detail view with overview, results, and learnings tabs"
```

---

## Task 8: CSS for New Components

**Files:**
- Modify: `web/static/css/style.css`

### Step 8.1: Append styles to `web/static/css/style.css`

Append to the end of `style.css`:

```css
/* ── Experiments Layout ───────────────────────────────────── */

.experiments-page { padding: 1.5rem; }

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
}

.page-header h1 { margin: 0; }

.header-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.view-toggle {
  display: flex;
  border: 1px solid var(--border, #ddd);
  border-radius: 6px;
  overflow: hidden;
}

.view-toggle button {
  padding: 0.4rem 0.9rem;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 0.875rem;
}

.view-toggle button.active {
  background: var(--accent, #3b82f6);
  color: white;
}

/* ── Kanban ───────────────────────────────────────────────── */

.kanban-board {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  align-items: start;
}

.kanban-col {
  background: var(--surface-alt, #f8f9fa);
  border-radius: 8px;
  padding: 0.75rem;
  min-height: 200px;
}

.kanban-col-header {
  display: flex;
  justify-content: space-between;
  font-weight: 600;
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted, #6b7280);
}

.kanban-col-header .count {
  background: var(--border, #ddd);
  border-radius: 999px;
  padding: 0 0.4rem;
  font-size: 0.75rem;
}

.kanban-card {
  background: white;
  border: 1px solid var(--border, #ddd);
  border-radius: 6px;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  cursor: pointer;
  transition: box-shadow 0.15s;
}

.kanban-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }

.card-name { font-weight: 500; margin-bottom: 0.25rem; }

.card-meta {
  font-size: 0.8rem;
  color: var(--text-muted, #6b7280);
  margin-bottom: 0.5rem;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.days { font-size: 0.75rem; color: var(--text-muted, #6b7280); }

.kanban-empty {
  text-align: center;
  color: var(--text-muted, #6b7280);
  padding: 1rem;
  font-size: 0.875rem;
}

/* ── Timeline ─────────────────────────────────────────────── */

.timeline-view { padding: 0.5rem 0; }

.timeline-controls { margin-bottom: 1rem; font-size: 0.875rem; }

.timeline-group { margin-bottom: 1.5rem; }

.timeline-group-label {
  font-weight: 600;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted, #6b7280);
  margin-bottom: 0.5rem;
}

.timeline-row {
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 0.5rem;
  align-items: center;
  margin-bottom: 0.3rem;
}

.timeline-row-label {
  font-size: 0.875rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.timeline-row-bar {
  position: relative;
  height: 24px;
  background: var(--surface-alt, #f3f4f6);
  border-radius: 4px;
  overflow: hidden;
}

.timeline-bar {
  position: absolute;
  top: 4px;
  height: 16px;
  border-radius: 3px;
  cursor: pointer;
  opacity: 0.85;
  transition: opacity 0.15s;
}

.timeline-bar:hover { opacity: 1; }

.timeline-bar.status-plan { background: #93c5fd; }
.timeline-bar.status-instrumentation { background: #fcd34d; }
.timeline-bar.status-running { background: #6ee7b7; }
.timeline-bar.status-completed { background: #9ca3af; }
.timeline-bar.status-paused { background: #fdba74; }
.timeline-bar.status-cancelled { background: #fca5a5; }

/* ── Badges ───────────────────────────────────────────────── */

.badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 500;
}

.badge-blue { background: #dbeafe; color: #1d4ed8; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-gray { background: #f3f4f6; color: #374151; }
.badge-orange { background: #ffedd5; color: #9a3412; }
.badge-red { background: #fee2e2; color: #991b1b; }

/* ── Detail Page ──────────────────────────────────────────── */

.detail-page { padding: 1.5rem; max-width: 900px; }

.breadcrumb { margin-bottom: 1rem; font-size: 0.875rem; }
.breadcrumb a { color: var(--accent, #3b82f6); text-decoration: none; }

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1.5rem;
}

.detail-header h1 { margin: 0 0 0.25rem; }

.detail-meta { font-size: 0.875rem; color: var(--text-muted, #6b7280); }

.detail-actions { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }

/* ── Tabs ─────────────────────────────────────────────────── */

.tabs {
  display: flex;
  gap: 0;
  border-bottom: 2px solid var(--border, #e5e7eb);
  margin-bottom: 1.5rem;
}

.tabs button {
  padding: 0.6rem 1.2rem;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 0.9rem;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  color: var(--text-muted, #6b7280);
}

.tabs button.active {
  border-bottom-color: var(--accent, #3b82f6);
  color: var(--accent, #3b82f6);
  font-weight: 600;
}

.tabs button .count {
  background: var(--surface-alt, #f3f4f6);
  border-radius: 999px;
  padding: 0.1rem 0.4rem;
  font-size: 0.75rem;
  margin-left: 0.3rem;
}

.tab-content { padding: 0.5rem 0; }
.tab-actions { margin-bottom: 1rem; }

/* ── Overview fields ──────────────────────────────────────── */

.overview-grid { display: flex; flex-direction: column; gap: 1rem; }
.field-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; }
.field-group label { display: block; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted, #6b7280); margin-bottom: 0.25rem; }
.field-group p { margin: 0; }

/* ── Result card ──────────────────────────────────────────── */

.result-card {
  border: 1px solid var(--border, #e5e7eb);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.75rem;
}

.result-header {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  margin-bottom: 0.75rem;
}

.calculator-type {
  font-weight: 600;
  font-size: 0.875rem;
  text-transform: capitalize;
}

.result-date { margin-left: auto; font-size: 0.8rem; color: var(--text-muted, #6b7280); }

.result-stats { display: flex; gap: 1.5rem; }
.result-stats .stat label { display: block; font-size: 0.75rem; color: var(--text-muted, #6b7280); }

/* ── Learning card ────────────────────────────────────────── */

.learning-card {
  border: 1px solid var(--border, #e5e7eb);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.75rem;
}

.learning-card p { margin: 0 0 0.5rem; }

.learning-meta { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
.tag { background: var(--surface-alt, #f3f4f6); border-radius: 4px; padding: 0.15rem 0.5rem; font-size: 0.75rem; }
.learning-date { font-size: 0.75rem; color: var(--text-muted, #6b7280); }

/* ── Modals ───────────────────────────────────────────────── */

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  background: white;
  border-radius: 10px;
  width: 520px;
  max-width: 95vw;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.25rem 1.5rem 0;
}

.modal-header h2 { margin: 0; font-size: 1.1rem; }
.close-btn { background: none; border: none; font-size: 1.5rem; cursor: pointer; color: var(--text-muted, #6b7280); }

.modal-body { padding: 1.25rem 1.5rem; }
.modal-footer { padding: 1rem 1.5rem; display: flex; justify-content: flex-end; gap: 0.75rem; border-top: 1px solid var(--border, #e5e7eb); }

/* ── Forms ────────────────────────────────────────────────── */

.form-group { display: flex; flex-direction: column; gap: 0.35rem; margin-bottom: 1rem; }
.form-group label { font-size: 0.875rem; font-weight: 500; }
.form-group input, .form-group select, .form-group textarea {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border, #d1d5db);
  border-radius: 6px;
  font-size: 0.875rem;
  font-family: inherit;
  width: 100%;
  box-sizing: border-box;
}

.form-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.75rem; }
.form-actions { display: flex; justify-content: flex-end; gap: 0.75rem; margin-top: 1rem; }
.hint { font-size: 0.8rem; color: var(--text-muted, #6b7280); margin: 0; }

/* ── Buttons ──────────────────────────────────────────────── */

.btn-primary {
  background: var(--accent, #3b82f6);
  color: white;
  border: none;
  border-radius: 6px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-size: 0.875rem;
  font-weight: 500;
}

.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.btn-secondary {
  background: transparent;
  color: var(--text, #374151);
  border: 1px solid var(--border, #d1d5db);
  border-radius: 6px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-size: 0.875rem;
}

/* ── Empty state ──────────────────────────────────────────── */

.empty-state {
  text-align: center;
  color: var(--text-muted, #6b7280);
  padding: 2rem;
  font-size: 0.9rem;
}

.empty-state a { color: var(--accent, #3b82f6); }

.loading { color: var(--text-muted, #6b7280); padding: 2rem; text-align: center; }
```

### Step 8.2: Full end-to-end test

```bash
cd /home/rabt/devel/vstats && v run web/ &
sleep 1
# Open http://localhost:8080/experiments
# Verify: page loads with 4-column Kanban, correct nav, no style errors
# Create experiment → appears as styled card in Plan column
# Click card → detail page loads with tabs
# Toggle to Timeline view → verify bars render
# Test all status transitions: plan → instrumentation → running → completed (with outcome)
# Verify modal requires outcome for "completed"
kill %1
```

### Step 8.3: Commit

```bash
git add web/static/css/style.css
git commit -m "feat(web): add CSS for kanban, timeline, detail view, and modals"
```

---

## Phase 1 Complete — Verification Checklist

- [ ] `v build web/` passes with no errors
- [ ] Server starts, `vstats.db` created on first run
- [ ] `POST /api/experiments` creates experiment, returns 201
- [ ] `GET /api/experiments` returns list
- [ ] `PUT /api/experiments/1` updates fields
- [ ] `PUT /api/experiments/1/status` transitions status, sets `started_at`/`completed_at`
- [ ] `DELETE /api/experiments/1` removes experiment and cascades to results/learnings
- [ ] `/experiments` shows Kanban with correct columns
- [ ] Timeline view renders horizontal bars color-coded by status
- [ ] Create modal creates experiment and it appears in Kanban
- [ ] Detail page loads at `/experiments/1`
- [ ] All three tabs render correct content
- [ ] Edit mode saves changes
- [ ] Status transitions work, completing requires outcome
- [ ] Learnings can be added with tags
- [ ] Nav shows 4 items, consistent across all pages
- [ ] Fonts, colors, spacing consistent with existing calculator pages

---

## Upcoming Phases

**Phase 2** (next plan): Design Wizard (4-step guided experiment creation) + Calculator integration ("Save to experiment" button, "Load from experiment" pre-fill)

**Phase 3** (follow-on): Learn section (decision trees, reference cards, glossary) + Dashboard (aggregated stats, throughput chart, learnings feed)
