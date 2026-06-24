# Experiment Wizard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 5-tab experiment modal with a 4-tab wizard (General / Setup / Config / Instrumentation) that persists to SQLite via the API and uses the metrics/populations catalogs for dropdowns.

**Architecture:** Backend gains 14 new columns on the `experiments` table and expanded request structs. The frontend replaces localStorage with API fetch calls, restructures the `expModal` Alpine component into 4 tabs with per-tab Save & Continue, and wires catalog dropdowns for metrics and populations.

**Tech Stack:** V + veb + SQLite ORM (backend); Alpine.js 3, vanilla CSS (frontend); all in `apps/tracker/`

## Global Constraints

- All backend code in `apps/tracker/main.v` — no new V files
- All frontend changes in `apps/tracker/public/app.js`, `index.html`, `app.css`
- `apps/` is git-ignored — no commits; verify by curl/browser
- Use `rtk proxy curl` for all HTTP verification
- Kill stale server instances with `kill $(lsof -ti:8080) 2>/dev/null; sleep 1` before starting fresh
- Start server: `cd /home/rabt/devel/vstats/apps/tracker && v run . > /tmp/tracker.log 2>&1 & sleep 6`
- Metrics and populations are loaded from the DB into `$store.app.metrics` and `$store.app.populations` on init (already implemented in prior tasks)
- Experiments persist to SQLite via `POST /api/experiments` (create) and `PUT /api/experiments/:id` (update) — localStorage is removed from the experiment flow
- `POST /api/metrics/:id/refresh` already exists and returns `{"value":"...","updated_at":"..."}` — used by Refresh now button
- The sidebar panel (summary/config/instrumentation tabs) is a separate component from the modal — both exist simultaneously; the sidebar shows read-only experiment data, the modal is for editing

---

### Task 1: Backend — 14 new columns + struct expansion + updated handlers

**Files:**
- Modify: `apps/tracker/main.v`

**Interfaces:**
- Produces:
  - `POST /api/experiments` body: all new fields accepted, returns created experiment with integer `id`
  - `PUT /api/experiments/:id` body: all new fields accepted, returns updated experiment
  - `GET /api/experiments` returns: all new fields included in each experiment object

- [ ] **Step 1: Add 14 ALTER TABLE migrations**

Find `run_migrations()` in `main.v` (around line 806). The `migrations` array ends with the `populations` CREATE TABLE line (currently the last entry). Append these 14 entries after it:

```v
'ALTER TABLE experiments ADD COLUMN description TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN owner TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN primary_metric_slug TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN population_slugs TEXT DEFAULT \'[]\'',
'ALTER TABLE experiments ADD COLUMN guardrail_slugs TEXT DEFAULT \'[]\'',
'ALTER TABLE experiments ADD COLUMN learning_metric_slugs TEXT DEFAULT \'[]\'',
'ALTER TABLE experiments ADD COLUMN flag_key TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN event_lineage TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN variants TEXT DEFAULT \'[]\'',
'ALTER TABLE experiments ADD COLUMN mde TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN duration_estimate TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN statistical_test TEXT DEFAULT \'\'',
'ALTER TABLE experiments ADD COLUMN power REAL DEFAULT 80',
'ALTER TABLE experiments ADD COLUMN significance_level REAL DEFAULT 0.05',
```

These run with `or {}` so they silently no-op if the column already exists.

- [ ] **Step 2: Add new fields to the `Experiment` struct**

The `Experiment` struct starts around line 24. Add the following fields after the existing `completed_at string` field:

```v
description           string
owner                 string
primary_metric_slug   string
population_slugs      string
guardrail_slugs       string
learning_metric_slugs string
flag_key              string
event_lineage         string
variants              string
mde                   string
duration_estimate     string
statistical_test      string
power                 f64
significance_level    f64
optimizer_baseline            f64
optimizer_daily_traffic       int
optimizer_min_relative_lift   f64
optimizer_prior_conviction    f64
optimizer_metric_std_dev      f64
optimizer_worth_running       int
```

Note: the 6 `optimizer_*` fields are added to the struct here so the ORM can use them naturally — they already exist as columns from prior migrations. Remove them from the raw SQL UPDATE workaround in `create_experiment`.

- [ ] **Step 3: Update `CreateExperimentReq` struct**

Find `CreateExperimentReq` (around line 650). Replace the entire struct with:

```v
struct CreateExperimentReq {
	name                          string
	description                   string
	owner                         string
	primary_metric_slug           string
	population_slugs              string
	guardrail_slugs               string
	learning_metric_slugs         string
	flag_key                      string
	event_lineage                 string
	variants                      string
	mde                           string
	duration_estimate             string
	statistical_test              string
	power                         f64
	significance_level            f64
	optimizer_baseline            f64
	optimizer_daily_traffic       int
	optimizer_min_relative_lift   f64
	optimizer_prior_conviction    f64
	optimizer_metric_std_dev      f64
	optimizer_worth_running       int
}
```

- [ ] **Step 4: Update `UpdateExperimentReq` struct**

Find `UpdateExperimentReq` (around line 674). Replace the entire struct with:

```v
struct UpdateExperimentReq {
	name                          string
	description                   string
	owner                         string
	primary_metric_slug           string
	population_slugs              string
	guardrail_slugs               string
	learning_metric_slugs         string
	flag_key                      string
	event_lineage                 string
	variants                      string
	mde                           string
	duration_estimate             string
	statistical_test              string
	status                        string
	outcome                       string
	notes                         string
	power                         f64
	significance_level            f64
	optimizer_baseline            f64
	optimizer_daily_traffic       int
	optimizer_min_relative_lift   f64
	optimizer_prior_conviction    f64
	optimizer_metric_std_dev      f64
	optimizer_worth_running       int
}
```

- [ ] **Step 5: Update `create_experiment` handler**

Find `create_experiment` handler (around line 116). Replace the `exp := Experiment{...}` literal and the raw SQL UPDATE with:

```v
@['/api/experiments'; post]
pub fn (app &App) create_experiment(mut ctx Context) veb.Result {
	req := json.decode(CreateExperimentReq, ctx.req.data) or {
		return ctx.request_error('invalid JSON')
	}
	now := get_current_time()
	exp := Experiment{
		name:                        req.name
		description:                 req.description
		owner:                       req.owner
		primary_metric_slug:         req.primary_metric_slug
		population_slugs:            req.population_slugs
		guardrail_slugs:             req.guardrail_slugs
		learning_metric_slugs:       req.learning_metric_slugs
		flag_key:                    req.flag_key
		event_lineage:               req.event_lineage
		variants:                    req.variants
		mde:                         req.mde
		duration_estimate:           req.duration_estimate
		statistical_test:            req.statistical_test
		power:                       req.power
		significance_level:          req.significance_level
		optimizer_baseline:          req.optimizer_baseline
		optimizer_daily_traffic:     req.optimizer_daily_traffic
		optimizer_min_relative_lift: req.optimizer_min_relative_lift
		optimizer_prior_conviction:  req.optimizer_prior_conviction
		optimizer_metric_std_dev:    req.optimizer_metric_std_dev
		optimizer_worth_running:     req.optimizer_worth_running
		status:     'plan'
		outcome:    ''
		notes:      ''
		created_at: now
		updated_at: now
		started_at: ''
		completed_at: ''
	}
	sql app.db {
		insert exp into Experiment
	} or {
		return ctx.server_error('failed to create experiment')
	}
	last_id := app.db.last_id()
	rows := app.db.exec('SELECT * FROM experiments WHERE id = ${last_id}') or {
		return ctx.server_error('fetch failed')
	}
	// Return the full experiment including the auto-assigned id
	created := sql app.db {
		select from Experiment where id == int(last_id)
	} or { return ctx.server_error('fetch failed') }
	if created.len == 0 { return ctx.server_error('not found after insert') }
	ctx.res.set_status(.created)
	return ctx.json(created[0])
}
```

- [ ] **Step 6: Update `update_experiment` handler**

Find `update_experiment` (around line 157). Replace the `sql app.db { update Experiment set ... }` block with one that includes all new fields:

```v
@['/api/experiments/:id'; put]
pub fn (app &App) update_experiment(mut ctx Context, id int) veb.Result {
	req := json.decode(UpdateExperimentReq, ctx.req.data) or {
		return ctx.request_error('invalid JSON')
	}
	now := get_current_time()
	sql app.db {
		update Experiment set
			name = req.name,
			description = req.description,
			owner = req.owner,
			primary_metric_slug = req.primary_metric_slug,
			population_slugs = req.population_slugs,
			guardrail_slugs = req.guardrail_slugs,
			learning_metric_slugs = req.learning_metric_slugs,
			flag_key = req.flag_key,
			event_lineage = req.event_lineage,
			variants = req.variants,
			mde = req.mde,
			duration_estimate = req.duration_estimate,
			statistical_test = req.statistical_test,
			power = req.power,
			significance_level = req.significance_level,
			status = req.status,
			outcome = req.outcome,
			notes = req.notes,
			optimizer_baseline = req.optimizer_baseline,
			optimizer_daily_traffic = req.optimizer_daily_traffic,
			optimizer_min_relative_lift = req.optimizer_min_relative_lift,
			optimizer_prior_conviction = req.optimizer_prior_conviction,
			optimizer_metric_std_dev = req.optimizer_metric_std_dev,
			optimizer_worth_running = req.optimizer_worth_running,
			updated_at = now
		where id == id
	} or {
		return ctx.server_error('failed to update experiment')
	}
	updated := sql app.db {
		select from Experiment where id == id
	} or { return ctx.server_error('failed to fetch experiment') }
	if updated.len == 0 { return ctx.not_found() }
	return ctx.json(updated[0])
}
```

- [ ] **Step 7: Restart server and verify all endpoints**

```bash
kill $(lsof -ti:8080) 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . > /tmp/tracker.log 2>&1 & sleep 6
```

```bash
# Create with new fields
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/experiments \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"Checkout CTA Test",
    "description":"Testing button color",
    "owner":"Sarah Chen",
    "primary_metric_slug":"checkout_conversion",
    "population_slugs":"[\"all_active_users\"]",
    "guardrail_slugs":"[]",
    "learning_metric_slugs":"[]",
    "flag_key":"checkout_cta_v2",
    "event_lineage":"click → checkout",
    "variants":"[{\"name\":\"Control\",\"traffic\":50},{\"name\":\"Treatment\",\"traffic\":50}]",
    "mde":"2.5%",
    "duration_estimate":"14 days",
    "statistical_test":"Two-proportion z-test",
    "power":80,
    "significance_level":0.05,
    "optimizer_baseline":0.41,
    "optimizer_daily_traffic":1370,
    "optimizer_min_relative_lift":0.05,
    "optimizer_prior_conviction":0.5,
    "optimizer_metric_std_dev":0.0,
    "optimizer_worth_running":1
  }'
# Expected: HTTP:201, response includes "id":N, "description":"Testing button color", "owner":"Sarah Chen"

# List includes new fields
rtk proxy curl -s http://localhost:8080/api/experiments | python3 -c "import sys,json; d=json.load(sys.stdin); e=d[0]; print(e.get('id'), e.get('owner'), e.get('primary_metric_slug'))"
# Expected: 1 Sarah Chen checkout_conversion

# Update
ID=$(rtk proxy curl -s http://localhost:8080/api/experiments | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X PUT http://localhost:8080/api/experiments/${ID} \
  -H 'Content-Type: application/json' \
  -d '{"name":"Checkout CTA Test","description":"Updated desc","owner":"Sarah Chen","primary_metric_slug":"checkout_conversion","population_slugs":"[]","guardrail_slugs":"[]","learning_metric_slugs":"[]","flag_key":"checkout_cta_v2","event_lineage":"","variants":"[]","mde":"","duration_estimate":"","statistical_test":"","status":"plan","outcome":"","notes":"","power":80,"significance_level":0.05,"optimizer_baseline":0,"optimizer_daily_traffic":0,"optimizer_min_relative_lift":0,"optimizer_prior_conviction":0,"optimizer_metric_std_dev":0,"optimizer_worth_running":-1}'
# Expected: HTTP:200, "description":"Updated desc"

# Delete
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X DELETE http://localhost:8080/api/experiments/${ID}
# Expected: HTTP:200 "deleted"
```

---

### Task 2: Frontend store — localStorage → API + form shape + tab names

**Files:**
- Modify: `apps/tracker/public/app.js`

**Interfaces:**
- Consumes: Task 1's `GET /api/experiments`, `POST /api/experiments`, `PUT /api/experiments/:id`, `DELETE /api/experiments/:id`
- Produces:
  - `Alpine.store('app').experiments` — loaded from API on init
  - `Alpine.store('app').saveExp(data)` — async, calls API, returns saved object with server-assigned id
  - `Alpine.store('app').deleteExp(id)` — async, calls API then removes from store
  - `expModal` Alpine.data component with new tab array, new form shape, `saveAndContinue()`, `saveAndClose()`, `buildPayload()`, `prevTab()`

- [ ] **Step 1: Replace localStorage experiment loading in store init()**

Find `init()` in the Alpine store (around line 320). Replace the localStorage experiment block:

```js
// REMOVE these lines:
let stored = null;
try { stored = JSON.parse(localStorage.getItem('exp_v2') || 'null'); } catch { stored = null; }
this.experiments = stored || JSON.parse(JSON.stringify(this.INITIAL_EXPERIMENTS));
this.view = localStorage.getItem('exp_v2_view') || 'board';
const selId = localStorage.getItem('exp_v2_sel');
if (selId) {
    const e = this.experiments.find(x => x.id === selId);
    if (e) this.selectedExp = e;
}
```

With:

```js
this.view = 'board';
fetch('/api/experiments').then(r => r.json()).then(data => {
    this.experiments = Array.isArray(data) ? data : [];
}).catch(() => { this.experiments = []; });
```

- [ ] **Step 2: Replace `persist()` — remove it**

Find `persist()` method (around line 341) and delete it entirely:

```js
// DELETE this method:
persist() {
    localStorage.setItem('exp_v2', JSON.stringify(this.experiments));
    localStorage.setItem('exp_v2_view', this.view);
    if (this.selectedExp) localStorage.setItem('exp_v2_sel', this.selectedExp.id);
    else localStorage.removeItem('exp_v2_sel');
},
```

- [ ] **Step 3: Replace `saveExp()` with async API version**

Find `saveExp(data)` (around line 357). Replace it:

```js
async saveExp(data) {
    const isNew = !data.id;
    const url = isNew ? '/api/experiments' : `/api/experiments/${data.id}`;
    const method = isNew ? 'POST' : 'PUT';
    const resp = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    if (!resp.ok) throw new Error('save failed: ' + resp.status);
    const saved = await resp.json();
    const idx = this.experiments.findIndex(x => String(x.id) === String(saved.id));
    if (idx >= 0) this.experiments[idx] = saved;
    else this.experiments.unshift(saved);
    if (this.selectedExp?.id === saved.id) this.selectedExp = saved;
    this.modalExp = null;
    return saved;
},
```

- [ ] **Step 4: Replace `deleteExp()` with async API version**

Find `deleteExp(id)` (around line 369). Replace it:

```js
async deleteExp(id) {
    if (!confirm('Delete this experiment? This cannot be undone.')) return;
    await fetch(`/api/experiments/${id}`, { method: 'DELETE' });
    this.experiments = this.experiments.filter(e => String(e.id) !== String(id));
    if (String(this.selectedExp?.id) === String(id)) this.selectedExp = null;
},
```

- [ ] **Step 5: Remove `persist()` calls from `selectExp` and `cloneExp`**

Find `selectExp(exp)` (around line 348) — remove `this.persist()` call from it.
Find `cloneExp(exp)` (around line 376) — this now needs to call `saveExp`:

```js
selectExp(exp) {
    this.selectedExp = this.selectedExp?.id === exp.id ? null : exp;
},

async cloneExp(exp) {
    const clone = {
        ...exp,
        id: undefined,
        name: `${exp.name} (Copy)`,
        status: 'plan',
        created_at: '',
        updated_at: '',
    };
    await this.saveExp(clone);
},
```

- [ ] **Step 6: Update `expModal` tab array and form initializer**

Find `Alpine.data('expModal', () => ({` (around line 422). Replace the `tab:` and `form:` initialization:

```js
Alpine.data('expModal', () => ({
    tab: 'General',
    form: {
        id: null,
        name: '', description: '', owner: '',
        primaryMetricSlug: '',
        populationSlugs: [],
        guardrailSlugs: [],
        learningMetricSlugs: [],
        optimizer: {
            baseline: null, dailyTraffic: null, minRelativeLift: 0.05,
            priorConviction: 0.50, metricStdDev: 0.0, maxDays: 90,
            calculating: false, result: null, overrideConfirmed: false,
        },
        mde: '', sampleSize: '', power: 80, significanceLevel: 0.05,
        statisticalTest: 'Two-proportion z-test', durationEstimate: '',
        variants: [
            { name: 'Control', description: '', traffic: 50 },
            { name: 'Treatment', description: '', traffic: 50 },
        ],
        flagKey: '',
        eventLineage: '',
        // management fields (not in wizard, editable from sidebar)
        status: 'plan', outcome: '', notes: '',
        started_at: '', completed_at: '',
    },
```

- [ ] **Step 7: Update `init()` and `$watch` in `expModal` to hydrate new fields**

Find the `init()` method inside `expModal` (around line 444). Replace it to hydrate the new form shape from an existing experiment:

```js
init() {
    this._hydrate(this.app.modalExp);
    this.$watch('app.modalExp', (val) => {
        this._hydrate(val);
        this.tab = 'General';
    });
},

_hydrate(src) {
    if (src && typeof src === 'object' && src.id) {
        this.form = {
            id: src.id,
            name: src.name || '',
            description: src.description || '',
            owner: src.owner || '',
            primaryMetricSlug: src.primary_metric_slug || '',
            populationSlugs: this._parseJson(src.population_slugs, []),
            guardrailSlugs: this._parseJson(src.guardrail_slugs, []),
            learningMetricSlugs: this._parseJson(src.learning_metric_slugs, []),
            optimizer: {
                baseline: src.optimizer_baseline || null,
                dailyTraffic: src.optimizer_daily_traffic || null,
                minRelativeLift: src.optimizer_min_relative_lift || 0.05,
                priorConviction: src.optimizer_prior_conviction || 0.50,
                metricStdDev: src.optimizer_metric_std_dev || 0.0,
                maxDays: src.optimizer_max_days || 90,
                calculating: false, result: null, overrideConfirmed: false,
            },
            mde: src.mde || '',
            sampleSize: src.sample_size_per_group || '',
            power: src.power || 80,
            significanceLevel: src.significance_level || 0.05,
            statisticalTest: src.statistical_test || 'Two-proportion z-test',
            durationEstimate: src.duration_estimate || '',
            variants: this._parseJson(src.variants, [
                { name: 'Control', description: '', traffic: 50 },
                { name: 'Treatment', description: '', traffic: 50 },
            ]),
            flagKey: src.flag_key || '',
            eventLineage: src.event_lineage || '',
            status: src.status || 'plan',
            outcome: src.outcome || '',
            notes: src.notes || '',
            started_at: src.started_at || '',
            completed_at: src.completed_at || '',
        };
    } else {
        this.form = {
            id: null,
            name: '', description: '', owner: '',
            primaryMetricSlug: '',
            populationSlugs: [],
            guardrailSlugs: [],
            learningMetricSlugs: [],
            optimizer: {
                baseline: null, dailyTraffic: null, minRelativeLift: 0.05,
                priorConviction: 0.50, metricStdDev: 0.0, maxDays: 90,
                calculating: false, result: null, overrideConfirmed: false,
            },
            mde: '', sampleSize: '', power: 80, significanceLevel: 0.05,
            statisticalTest: 'Two-proportion z-test', durationEstimate: '',
            variants: [
                { name: 'Control', description: '', traffic: 50 },
                { name: 'Treatment', description: '', traffic: 50 },
            ],
            flagKey: '', eventLineage: '',
            status: 'plan', outcome: '', notes: '',
            started_at: '', completed_at: '',
        };
    }
},

_parseJson(val, fallback) {
    if (!val) return fallback;
    if (Array.isArray(val)) return val;
    try { return JSON.parse(val); } catch { return fallback; }
},
```

- [ ] **Step 8: Add `buildPayload()`, `saveAndContinue()`, `saveAndClose()`, `prevTab()` methods**

Add these methods to the `expModal` component, replacing the old `save()` method:

```js
buildPayload() {
    const f = this.form;
    const o = f.optimizer;
    return {
        id: f.id || undefined,
        name: f.name,
        description: f.description,
        owner: f.owner,
        primary_metric_slug: f.primaryMetricSlug,
        population_slugs: JSON.stringify(f.populationSlugs),
        guardrail_slugs: JSON.stringify(f.guardrailSlugs),
        learning_metric_slugs: JSON.stringify(f.learningMetricSlugs),
        flag_key: f.flagKey,
        event_lineage: f.eventLineage,
        variants: JSON.stringify(f.variants),
        mde: f.mde,
        duration_estimate: f.durationEstimate,
        statistical_test: f.statisticalTest,
        power: f.power,
        significance_level: f.significanceLevel,
        status: f.status,
        outcome: f.outcome,
        notes: f.notes,
        optimizer_baseline: o.baseline || 0,
        optimizer_daily_traffic: o.dailyTraffic || 0,
        optimizer_min_relative_lift: o.minRelativeLift || 0,
        optimizer_prior_conviction: o.priorConviction || 0,
        optimizer_metric_std_dev: o.metricStdDev || 0,
        optimizer_worth_running: o.result
            ? (o.result.worth_running ? 1 : (o.overrideConfirmed ? 0 : -1))
            : -1,
    };
},

async saveAndContinue() {
    if (this.tab === 'General') {
        if (!this.form.name.trim() || !this.form.owner.trim()) {
            alert('Name and Owner are required.');
            return;
        }
    }
    try {
        const saved = await this.app.saveExp(this.buildPayload());
        // saveExp sets modalExp = null, but we want to stay in the wizard
        this.app.modalExp = saved;
        this.form.id = saved.id;
        this.nextTab();
    } catch (e) {
        alert('Save failed. Please try again.');
    }
},

async saveAndClose() {
    if (this.tab === 'General') {
        if (!this.form.name.trim() || !this.form.owner.trim()) {
            alert('Name and Owner are required.');
            return;
        }
    }
    try {
        await this.app.saveExp(this.buildPayload());
        // saveExp already sets modalExp = null, closing the modal
    } catch (e) {
        alert('Save failed. Please try again.');
    }
},

nextTab() {
    const tabs = ['General', 'Setup', 'Config', 'Instrumentation'];
    const i = tabs.indexOf(this.tab);
    if (i < tabs.length - 1) this.tab = tabs[i + 1];
},

prevTab() {
    const tabs = ['General', 'Setup', 'Config', 'Instrumentation'];
    const i = tabs.indexOf(this.tab);
    if (i > 0) this.tab = tabs[i - 1];
},
```

Note: `saveExp()` sets `this.modalExp = null` to close the modal. `saveAndContinue()` reopens it immediately with the saved experiment so the wizard can continue. This avoids changing `saveExp()`'s contract.

- [ ] **Step 9: Restart server and verify store loads from API**

```bash
kill $(lsof -ti:8080) 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . > /tmp/tracker.log 2>&1 & sleep 6
```

First create a test experiment via curl:
```bash
rtk proxy curl -s -X POST http://localhost:8080/api/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"API Load Test","description":"Verifying API load","owner":"Test User","primary_metric_slug":"","population_slugs":"[]","guardrail_slugs":"[]","learning_metric_slugs":"[]","flag_key":"","event_lineage":"","variants":"[]","mde":"","duration_estimate":"","statistical_test":"","status":"plan","outcome":"","notes":"","power":80,"significance_level":0.05,"optimizer_baseline":0,"optimizer_daily_traffic":0,"optimizer_min_relative_lift":0,"optimizer_prior_conviction":0,"optimizer_metric_std_dev":0,"optimizer_worth_running":-1}'
```

Then verify the app loads it (check app.js doesn't reference localStorage for experiments):
```bash
rtk proxy curl -s http://localhost:8080/app.js | grep -c "localStorage"
# Expected: 0 (or close — view preference localStorage is OK to keep if present)

rtk proxy curl -s http://localhost:8080/api/experiments | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d), 'experiments, first:', d[0]['name'] if d else 'none')"
# Expected: 1 experiments, first: API Load Test
```

---

### Task 3: General + Setup tab HTML, CSS, and Alpine methods

**Files:**
- Modify: `apps/tracker/public/index.html`
- Modify: `apps/tracker/public/app.js`
- Modify: `apps/tracker/public/app.css`

**Interfaces:**
- Consumes: Task 2's `expModal` form shape, `$store.app.metrics`, `$store.app.populations`, `POST /api/metrics/:id/refresh`
- Produces:
  - General tab HTML with name/description/owner fields
  - Setup tab HTML with primary metric dropdown, baseline + refresh button, 3 multi-select catalog lists, optimizer params, calculate button, result card
  - `refreshBaseline()`, `togglePopulation()`, `toggleGuardrail()`, `toggleLearningMetric()`, `get selectedMetricId()`, `propagateOptimizerResult()` methods in `expModal`
  - Updated `runOptimizer()` to call `propagateOptimizerResult()`
  - CSS for catalog multi-select lists

- [ ] **Step 1: Add Alpine methods for Setup tab**

In `app.js`, find the `expModal` component and add these methods after `prevTab()`:

```js
get selectedMetricId() {
    const m = this.app.metrics.find(x => x.slug === this.form.primaryMetricSlug);
    return m ? m.id : null;
},

async refreshBaseline() {
    const id = this.selectedMetricId;
    if (!id) return;
    this._refreshingBaseline = true;
    try {
        const resp = await fetch(`/api/metrics/${id}/refresh`, { method: 'POST' });
        if (resp.ok) {
            const data = await resp.json();
            this.form.optimizer.baseline = parseFloat(data.value) || null;
        }
    } finally {
        this._refreshingBaseline = false;
    }
},

togglePopulation(slug) {
    const i = this.form.populationSlugs.indexOf(slug);
    if (i >= 0) this.form.populationSlugs.splice(i, 1);
    else this.form.populationSlugs.push(slug);
},

toggleGuardrail(slug) {
    const i = this.form.guardrailSlugs.indexOf(slug);
    if (i >= 0) this.form.guardrailSlugs.splice(i, 1);
    else this.form.guardrailSlugs.push(slug);
},

toggleLearningMetric(slug) {
    const i = this.form.learningMetricSlugs.indexOf(slug);
    if (i >= 0) this.form.learningMetricSlugs.splice(i, 1);
    else this.form.learningMetricSlugs.push(slug);
},

propagateOptimizerResult(result) {
    if (!result || !result.worth_running) return;
    this.form.mde = result.mde_absolute ? result.mde_absolute.toFixed(4) : this.form.mde;
    this.form.sampleSize = result.sample_size_per_variant || this.form.sampleSize;
    this.form.durationEstimate = result.optimal_days ? `${result.optimal_days} days` : this.form.durationEstimate;
},
```

Also add `_refreshingBaseline: false` to the form initializer (alongside `form:`).

- [ ] **Step 2: Update `runOptimizer()` to call `propagateOptimizerResult`**

Find `runOptimizer()` in `expModal` (around line 512). After the optimizer result is received and assigned to `this.form.optimizer.result`, add a call:

```js
this.propagateOptimizerResult(data);
```

The exact placement is after `this.form.optimizer.result = data;` and before any error handling.

- [ ] **Step 3: Add CSS for catalog multi-select lists**

Append to the end of `apps/tracker/public/app.css`:

```css
/* Wizard catalog multi-select */
.catalog-list { display:flex; flex-direction:column; gap:4px; max-height:160px; overflow-y:auto; border:1.5px solid #E5E7EB; border-radius:8px; padding:6px; }
.catalog-item { display:flex; align-items:center; gap:8px; padding:5px 8px; border-radius:6px; cursor:pointer; font-size:13px; color:#374151; user-select:none; }
.catalog-item:hover { background:#F3F4F6; }
.catalog-item.selected { background:#EFF6FF; color:#1D4ED8; }
.catalog-item-name { flex:1; }
.catalog-item-meta { font-size:11px; color:#9CA3AF; }
.catalog-empty { font-size:12px; color:#9CA3AF; padding:8px; text-align:center; }
.baseline-row { display:flex; align-items:center; gap:8px; }
.baseline-row .form-input { flex:1; }
.btn-refresh-now { padding:6px 12px; border:none; border-radius:6px; background:#EFF6FF; color:#1D4ED8; font-size:12px; font-weight:600; cursor:pointer; white-space:nowrap; }
.btn-refresh-now:hover { background:#DBEAFE; }
.btn-refresh-now:disabled { opacity:0.5; cursor:default; }
```

- [ ] **Step 4: Replace the modal tab strip in index.html**

Find the `<div class="modal-tabs">` block (around line 723):

```html
<div class="modal-tabs">
    <template x-for="t in ['Basic','Summary','Design','Config','Instrumentation']" :key="t">
        <button class="modal-tab" :class="{ active: tab === t }" @click="tab = t" x-text="t"></button>
    </template>
</div>
```

Replace with:

```html
<div class="modal-tabs">
    <template x-for="t in ['General','Setup','Config','Instrumentation']" :key="t">
        <button class="modal-tab" :class="{ active: tab === t }" @click="tab = t" x-text="t"></button>
    </template>
</div>
```

- [ ] **Step 5: Replace General tab HTML (was "Basic")**

Find `<template x-if="tab === 'Basic'">` (around line 730) and replace the entire block including closing `</template>` with:

```html
<template x-if="tab === 'General'">
    <div>
        <div class="ff">
            <label class="ff-label">Experiment Name<span class="ff-required">*</span></label>
            <input class="form-input" x-model="form.name" placeholder="e.g. Checkout CTA Button Color"
                @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
        </div>
        <div class="ff">
            <label class="ff-label">Description</label>
            <textarea class="form-textarea" x-model="form.description" placeholder="What are you testing and why?" rows="3"
                @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'"></textarea>
        </div>
        <div class="ff">
            <label class="ff-label">Owner<span class="ff-required">*</span></label>
            <input class="form-input" x-model="form.owner" placeholder="e.g. Sarah Chen"
                @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
        </div>
    </div>
</template>
```

- [ ] **Step 6: Replace Summary tab HTML (now merged into Setup)**

Find `<template x-if="tab === 'Summary'">` and its closing `</template>`. Delete the entire block (it is fully replaced by the Setup tab in the next step).

- [ ] **Step 7: Replace Design tab HTML with new Setup tab**

Find `<template x-if="tab === 'Design'">` and its closing `</template>`. Replace the entire block with the new Setup tab:

```html
<template x-if="tab === 'Setup'">
    <div>
        <div class="ff">
            <label class="ff-label">Primary metric</label>
            <select class="form-select" x-model="form.primaryMetricSlug"
                @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
                <option value="">— select metric —</option>
                <template x-for="m in $store.app.metrics" :key="m.slug">
                    <option :value="m.slug" x-text="m.name"></option>
                </template>
            </select>
        </div>
        <div class="ff">
            <label class="ff-label">Baseline</label>
            <div class="baseline-row">
                <input type="number" step="any" class="form-input" x-model.number="form.optimizer.baseline"
                    placeholder="e.g. 0.41 for 41% CVR"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
                <button class="btn-refresh-now" :disabled="_refreshingBaseline || !selectedMetricId"
                    @click="refreshBaseline()">
                    <span x-show="!_refreshingBaseline">Refresh now</span>
                    <span x-show="_refreshingBaseline">Refreshing…</span>
                </button>
            </div>
        </div>
        <div class="ff">
            <label class="ff-label">Eligible populations</label>
            <template x-if="$store.app.populations.length === 0">
                <div class="catalog-empty">No populations defined yet — add them in the Populations catalog.</div>
            </template>
            <template x-if="$store.app.populations.length > 0">
                <div class="catalog-list">
                    <template x-for="p in $store.app.populations" :key="p.slug">
                        <div class="catalog-item" :class="{ selected: form.populationSlugs.includes(p.slug) }"
                            @click="togglePopulation(p.slug)">
                            <span class="catalog-item-name" x-text="p.name"></span>
                            <span class="catalog-item-meta" x-text="p.population_count ? Number(p.population_count).toLocaleString() : ''"></span>
                        </div>
                    </template>
                </div>
            </template>
        </div>
        <div class="ff">
            <label class="ff-label">Guardrail metrics</label>
            <template x-if="$store.app.metrics.length === 0">
                <div class="catalog-empty">No metrics defined yet.</div>
            </template>
            <template x-if="$store.app.metrics.length > 0">
                <div class="catalog-list">
                    <template x-for="m in $store.app.metrics" :key="m.slug">
                        <div class="catalog-item" :class="{ selected: form.guardrailSlugs.includes(m.slug) }"
                            @click="toggleGuardrail(m.slug)">
                            <span class="catalog-item-name" x-text="m.name"></span>
                        </div>
                    </template>
                </div>
            </template>
        </div>
        <div class="ff">
            <label class="ff-label">Learning metrics</label>
            <template x-if="$store.app.metrics.length > 0">
                <div class="catalog-list">
                    <template x-for="m in $store.app.metrics" :key="m.slug">
                        <div class="catalog-item" :class="{ selected: form.learningMetricSlugs.includes(m.slug) }"
                            @click="toggleLearningMetric(m.slug)">
                            <span class="catalog-item-name" x-text="m.name"></span>
                        </div>
                    </template>
                </div>
            </template>
        </div>
        <div class="form-grid-2" style="margin-top:8px">
            <div class="ff">
                <label class="ff-label">Daily users per variant</label>
                <input type="number" class="form-input" x-model.number="form.optimizer.dailyTraffic"
                    placeholder="e.g. 1370"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Minimum relative lift</label>
                <input type="number" step="0.01" class="form-input" x-model.number="form.optimizer.minRelativeLift"
                    placeholder="e.g. 0.05 = detect ≥5% improvement"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Prior conviction (0–1)</label>
                <input type="number" step="0.05" min="0" max="1" class="form-input" x-model.number="form.optimizer.priorConviction"
                    placeholder="0.2 skeptical · 0.5 moderate · 0.8 confident"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Metric std dev <span style="color:#9CA3AF;font-weight:400">(continuous only)</span></label>
                <input type="number" step="any" class="form-input" x-model.number="form.optimizer.metricStdDev"
                    placeholder="Leave 0 for proportions"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Max experiment days</label>
                <input type="number" class="form-input" x-model.number="form.optimizer.maxDays"
                    placeholder="90"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;margin:16px 0">
            <button style="min-width:120px;padding:8px 18px;border:none;border-radius:8px;background:#3B82F6;font-size:13px;font-weight:600;color:#fff;cursor:pointer"
                :style="form.optimizer.calculating ? 'opacity:0.6;cursor:not-allowed' : ''"
                :disabled="form.optimizer.calculating"
                @click="runOptimizer()">
                <span x-show="!form.optimizer.calculating">Calculate</span>
                <span x-show="form.optimizer.calculating">Running…</span>
            </button>
        </div>
        <template x-if="form.optimizer.result">
            <div>
                <template x-if="form.optimizer.result.worth_running">
                    <div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;padding:14px 16px;margin-bottom:14px">
                        <div style="font-weight:600;color:#166534;margin-bottom:8px">Recommended: GO</div>
                        <table style="font-size:13px;border-collapse:collapse;width:100%">
                            <tr><td style="color:#6B7280;padding:2px 8px 2px 0">Worth running</td><td style="font-weight:500">YES</td></tr>
                            <tr><td style="color:#6B7280;padding:2px 8px 2px 0">Recommended runtime</td><td style="font-weight:500" x-text="form.optimizer.result.optimal_days + ' days'"></td></tr>
                            <tr><td style="color:#6B7280;padding:2px 8px 2px 0">Power at optimal</td><td style="font-weight:500" x-text="(form.optimizer.result.power_at_optimal * 100).toFixed(1) + '%'"></td></tr>
                            <tr><td style="color:#6B7280;padding:2px 8px 2px 0">Monthly detection rate</td><td style="font-weight:500" x-text="form.optimizer.result.monthly_detection_rate.toFixed(3)"></td></tr>
                            <tr><td style="color:#6B7280;padding:2px 8px 2px 0">MDE (absolute)</td><td style="font-weight:500" x-text="form.optimizer.result.mde_absolute.toFixed(4)"></td></tr>
                        </table>
                    </div>
                </template>
                <template x-if="!form.optimizer.result.worth_running">
                    <div style="background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;padding:14px 16px;margin-bottom:14px">
                        <div style="font-weight:600;color:#92400E;margin-bottom:6px">Not recommended: NO</div>
                        <div style="font-size:13px;color:#78350F;margin-bottom:12px" x-text="form.optimizer.result.no_go_reason"></div>
                        <label style="display:flex;align-items:flex-start;gap:8px;cursor:pointer;font-size:13px;color:#92400E">
                            <input type="checkbox" style="margin-top:2px;flex-shrink:0" x-model="form.optimizer.overrideConfirmed">
                            <span>I understand this experiment is underpowered — proceed anyway</span>
                        </label>
                    </div>
                </template>
            </div>
        </template>
    </div>
</template>
```

- [ ] **Step 8: Verify General and Setup tabs in browser**

```bash
kill $(lsof -ti:8080) 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . > /tmp/tracker.log 2>&1 & sleep 6
```

```bash
# Verify HTML contains new tab structure
rtk proxy curl -s http://localhost:8080/ | grep -c "tab === 'General'"
# Expected: 1

rtk proxy curl -s http://localhost:8080/ | grep -c "tab === 'Setup'"
# Expected: 1

rtk proxy curl -s http://localhost:8080/app.css | grep -c "catalog-list"
# Expected: 1

# Verify old tabs are gone
rtk proxy curl -s http://localhost:8080/ | grep -c "tab === 'Basic'"
# Expected: 0

rtk proxy curl -s http://localhost:8080/ | grep -c "tab === 'Summary'"
# Expected: 0
```

---

### Task 4: Config + Instrumentation tabs + modal footer + sidebar updates

**Files:**
- Modify: `apps/tracker/public/index.html`

**Interfaces:**
- Consumes: Task 2's `expModal` form shape and methods (`saveAndContinue`, `saveAndClose`, `prevTab`)
- Produces:
  - Config tab HTML (MDE, sample size, power, significance, statistical test, duration, variants)
  - Instrumentation tab HTML (flag key, event lineage)
  - Modal footer with Back/Close/Save & Continue/Save & Close
  - Sidebar summary updated to show catalog-resolved names

- [ ] **Step 1: Replace Config tab HTML**

Find `<template x-if="tab === 'Config'">` (around line 920) and its closing `</template>`. Replace the entire block:

```html
<template x-if="tab === 'Config'">
    <div>
        <div class="form-grid-2">
            <div class="ff">
                <label class="ff-label">Min. Detectable Effect</label>
                <input class="form-input" x-model="form.mde" placeholder="e.g. 2.5%"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Sample Size (per variant)</label>
                <input type="number" class="form-input" x-model="form.sampleSize" placeholder="e.g. 15000"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Statistical Power (%)</label>
                <input type="number" class="form-input" x-model="form.power" placeholder="e.g. 80"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Significance Level (α)</label>
                <input type="number" class="form-input" x-model="form.significanceLevel" placeholder="e.g. 0.05"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
            <div class="ff">
                <label class="ff-label">Statistical Test</label>
                <select class="form-select" x-model="form.statisticalTest"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
                    <template x-for="s in $store.app.STAT_TESTS" :key="s">
                        <option :value="s" x-text="s || '—'"></option>
                    </template>
                </select>
            </div>
            <div class="ff">
                <label class="ff-label">Estimated Duration</label>
                <input class="form-input" x-model="form.durationEstimate" placeholder="e.g. 14 days"
                    @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
            </div>
        </div>
        <div style="margin-bottom:14px">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
                <label class="ff-label" style="margin-bottom:0">Variants</label>
                <button class="btn-add-variant" @click="addVariant()">+ Add Variant</button>
            </div>
            <template x-for="(v, i) in form.variants" :key="i">
                <div style="display:grid;grid-template-columns:1fr 1fr auto;gap:8px;margin-bottom:8px;align-items:center">
                    <input class="form-input form-input-sm" :value="v.name" @input="updateVariant(i, 'name', $el.value)" placeholder="Name"
                        @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
                    <input class="form-input form-input-sm" :value="v.description" @input="updateVariant(i, 'description', $el.value)" placeholder="Description"
                        @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
                    <div style="display:flex;gap:6px;align-items:center">
                        <input type="number" class="form-input form-input-narrow" :value="v.traffic" min="0" max="100"
                            @input="updateVariant(i, 'traffic', +$el.value)"
                            @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
                        <template x-if="form.variants.length > 2">
                            <button class="variant-remove-btn" @click="removeVariant(i)">
                                <span x-html="$store.app.Icons.Trash()"></span>
                            </button>
                        </template>
                    </div>
                </div>
            </template>
        </div>
    </div>
</template>
```

- [ ] **Step 2: Replace Instrumentation tab HTML**

Find `<template x-if="tab === 'Instrumentation'">` (around line 985) and its closing `</template>`. Replace the entire block:

```html
<template x-if="tab === 'Instrumentation'">
    <div>
        <div class="ff">
            <label class="ff-label">Flag Key</label>
            <input class="form-input form-mono" x-model="form.flagKey" placeholder="e.g. checkout_cta_v2"
                @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'">
        </div>
        <div class="ff">
            <label class="ff-label">Event Lineage</label>
            <textarea class="form-textarea" x-model="form.eventLineage" placeholder="event_a → event_b → conversion" rows="3"
                @focus="$el.style.borderColor='#3B82F6'" @blur="$el.style.borderColor='#E5E7EB'"></textarea>
        </div>
    </div>
</template>
```

- [ ] **Step 3: Replace modal footer with wizard navigation**

Find `<div class="modal-footer">` (around line 1029). Replace the entire footer block:

```html
<div class="modal-footer">
    <template x-if="tab !== 'General'">
        <button class="btn-cancel" @click="prevTab()">← Back</button>
    </template>
    <div style="flex:1"></div>
    <button class="btn-cancel" @click="$store.app.closeModal()">Close</button>
    <template x-if="tab !== 'Instrumentation'">
        <button class="btn-save" @click="saveAndContinue()">Save & Continue →</button>
    </template>
    <template x-if="tab === 'Instrumentation'">
        <button class="btn-save" @click="saveAndClose()">Save & Close</button>
    </template>
</div>
```

- [ ] **Step 4: Update sidebar summary tab to show catalog-resolved names**

The sidebar summary tab (around line 458–640) currently displays `targetPopulation`, `mainMetrics`, `guardrails`, `hypothesis`, `expectedResults`. Find and update these sections:

Find the sidebar section that shows `selectedExp.hypothesis` and `selectedExp.expectedResults` and remove those display blocks.

Find the section showing `selectedExp.targetPopulation` (or `mainMetrics`) and replace with:

```html
<template x-if="$store.app.selectedExp.primary_metric_slug">
    <div>
        <div class="sec-title" style="margin-bottom:4px">Primary Metric</div>
        <div class="sidebar-value" x-text="($store.app.metrics.find(m => m.slug === $store.app.selectedExp.primary_metric_slug) || {}).name || $store.app.selectedExp.primary_metric_slug"></div>
    </div>
</template>
<template x-if="($store.app.selectedExp.population_slugs || '[]') !== '[]'">
    <div>
        <div class="sec-title" style="margin-bottom:4px">Populations</div>
        <template x-for="slug in JSON.parse($store.app.selectedExp.population_slugs || '[]')" :key="slug">
            <div class="sidebar-value" x-text="($store.app.populations.find(p => p.slug === slug) || {}).name || slug"></div>
        </template>
    </div>
</template>
```

- [ ] **Step 5: End-to-end smoke test**

```bash
kill $(lsof -ti:8080) 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . > /tmp/tracker.log 2>&1 & sleep 6
```

```bash
# Verify Config and Instrumentation tabs present
rtk proxy curl -s http://localhost:8080/ | grep -c "tab === 'Config'"
# Expected: 1

rtk proxy curl -s http://localhost:8080/ | grep -c "tab === 'Instrumentation'"
# Expected: 1

# Verify old Design/Summary/Basic tabs gone
rtk proxy curl -s http://localhost:8080/ | grep -c "tab === 'Design'"
# Expected: 0

# Verify wizard footer buttons
rtk proxy curl -s http://localhost:8080/ | grep -c "saveAndContinue"
# Expected: 1

rtk proxy curl -s http://localhost:8080/ | grep -c "saveAndClose"
# Expected: 1

# Verify Save & Close text
rtk proxy curl -s http://localhost:8080/ | grep -c "Save & Close"
# Expected: 1

# Full API round-trip: create via wizard payload
rtk proxy curl -s -w "\nHTTP:%{http_code}" -X POST http://localhost:8080/api/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"Wizard E2E","description":"Full flow test","owner":"QA Bot","primary_metric_slug":"revenue_per_user","population_slugs":"[\"all_active_users\"]","guardrail_slugs":"[]","learning_metric_slugs":"[]","flag_key":"wizard_e2e","event_lineage":"click → purchase","variants":"[{\"name\":\"Control\",\"traffic\":50},{\"name\":\"Treatment\",\"traffic\":50}]","mde":"3%","duration_estimate":"21 days","statistical_test":"Two-proportion z-test","status":"plan","outcome":"","notes":"","power":80,"significance_level":0.05,"optimizer_baseline":0.12,"optimizer_daily_traffic":2000,"optimizer_min_relative_lift":0.05,"optimizer_prior_conviction":0.5,"optimizer_metric_std_dev":0.0,"optimizer_worth_running":1}'
# Expected: HTTP:201 with full experiment JSON including id

# Board still loads at /
rtk proxy curl -s -o /dev/null -w "HTTP:%{http_code}" http://localhost:8080/
# Expected: HTTP:200
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| 4-tab wizard General/Setup/Config/Instrumentation | Task 3 Steps 4-7, Task 4 Steps 1-2 |
| Back button (left, hidden on General) | Task 4 Step 3 |
| Close button always visible | Task 4 Step 3 |
| Save & Continue (all tabs except last) | Task 2 Step 8, Task 4 Step 3 |
| Save & Close (Instrumentation only) | Task 2 Step 8, Task 4 Step 3 |
| General: name + description + owner only | Task 3 Step 5 |
| Save & Continue on General validates name+owner | Task 2 Step 8 (saveAndContinue) |
| Setup: primary metric dropdown from catalog | Task 3 Step 7 |
| Setup: Baseline field renamed, Refresh now button | Task 3 Steps 1+7 |
| Refresh now calls POST /api/metrics/:id/refresh | Task 3 Step 1 (refreshBaseline) |
| Setup: populations multi-select from catalog with count | Task 3 Step 7 |
| Setup: guardrails multi-select from metrics catalog | Task 3 Step 7 |
| Setup: learning metrics multi-select from catalog | Task 3 Step 7 |
| Setup: optimizer params (5 fields unchanged) | Task 3 Step 7 |
| Setup: Calculate propagates to Config | Task 3 Steps 1+2 (propagateOptimizerResult) |
| No "Enter stats manually" link | Task 3 Step 7 (not included) |
| No hypothesis, expected results | Task 3 Steps 6+7 (Summary tab deleted) |
| Config: MDE/sample/power/significance overridable | Task 4 Step 1 |
| Config: statistical test dropdown | Task 4 Step 1 |
| Config: duration estimate | Task 4 Step 1 |
| Config: variants grid with add/remove | Task 4 Step 1 |
| Instrumentation: flag key + event lineage only | Task 4 Step 2 |
| No tracking tools / flagging system in wizard | Task 4 Step 2 (not included) |
| Experiments persisted to SQLite via API | Task 1 (backend), Task 2 Steps 1-5 |
| localStorage removed from experiment flow | Task 2 Steps 1-2 |
| 14 new columns with migrations | Task 1 Steps 1-2 |
| CreateExperimentReq + UpdateExperimentReq expanded | Task 1 Steps 3-4 |
| Sidebar shows resolved metric/population names | Task 4 Step 4 |
| Sidebar removes hypothesis/expected results display | Task 4 Step 4 |
| CSS for catalog multi-select lists | Task 3 Step 3 |
