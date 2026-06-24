# Design: Experiment Wizard Redesign

**Date:** 2026-06-24
**Scope:** `apps/tracker` — replace the 5-tab experiment modal with a 4-tab wizard; wire primary metric, populations, guardrails, and learning metrics to their catalogs
**Status:** Approved

---

## Problem

The current experiment creation modal has 5 tabs (Basic, Summary, Design, Config, Instrumentation) that overlap in purpose and mix identity fields with analytical configuration. The free-text metrics and population fields are disconnected from the Metrics and Populations catalogs already built. The single "Save" button at the bottom means the entire form must be completed before anything persists.

---

## Decision Summary

| Question | Decision | Rationale |
|---|---|---|
| Tab structure | General / Setup / Config / Instrumentation (4 tabs) | Eliminates redundant Summary tab; merges Design into Setup |
| Save model | Save & Continue persists per-tab | Allows abandoning mid-wizard without losing prior tabs; experiment created after General |
| Metrics / populations | Catalog dropdowns (replace free text) | Catalogs already exist; slugs are stable foreign-key-style references |
| Optimizer propagation | Calculate → auto-fills Config fields | Config values remain manually overridable |
| Removed fields | Hypothesis, expected results, target population (text), "Enter stats manually" | Hypothesis is derived from optimizer; population replaced by multi-select |
| Status / type / dates | Not in wizard; editable from sidebar only | Keeps General minimal; board state managed post-creation |

---

## Wizard Structure

### Navigation buttons

| Tab | Left | Right |
|---|---|---|
| General | — | Close · Save & Continue → |
| Setup | ← Back | Close · Save & Continue → |
| Config | ← Back | Close · Save & Continue → |
| Instrumentation | ← Back | Close · Save & Close |

- **Save & Continue**: calls `POST /api/experiments` (first save from General, when `form.id` is unset) or `PUT /api/experiments/:id` (subsequent tabs), then advances to next tab on success
- **Close**: closes the modal; unsaved changes on the current tab are lost, but previously saved tabs remain in the database
- **Save & Close**: same as Save & Continue but closes the modal instead of advancing

---

## Tab 1 — General

**Fields:**
- Name (text, required)
- Description (textarea, optional)
- Owner (text, required)

**Removed from this tab:** area/team (future Owners CRUD), status, type, result, start date, end date. These remain editable from the sidebar panel after creation. New experiments default to `status='planning'`, no dates.

**Save & Continue behavior:** Validates name and owner are non-empty. If `form.id` is unset, calls `POST /api/experiments` and stores the returned integer id in `form.id`. All subsequent tabs call `PUT /api/experiments/:id`. Advances to Setup on success.

---

## Tab 2 — Setup

Merges the current Summary and Design tabs.

**Fields in order:**

1. **Primary metric** — single `<select>` populated from `$store.app.metrics` (name → slug). Selection writes `form.primaryMetricSlug`. On change: auto-fills Baseline from the metric's `baseline_value` if available.

2. **Baseline** — number input bound to `form.optimizer.baseline`. Accompanied by a **Refresh now** button: calls `POST /api/metrics/:id/refresh` (where id is the selected metric's id), then writes the returned `value` into `form.optimizer.baseline`.

3. **Eligible populations** — multi-select checkbox list from `$store.app.populations`. Each row shows the population name and its last known count (`population_count` from the store). Selection writes `form.populationSlugs` (array of slugs).

4. **Guardrails** — multi-select checkbox list from `$store.app.metrics`. Writes `form.guardrailSlugs`.

5. **Learning metrics** — multi-select checkbox list from `$store.app.metrics`. Writes `form.learningMetricSlugs`.

6. **Optimizer parameters** (unchanged from current Design tab):
   - Daily users per variant → `form.optimizer.dailyTraffic`
   - Minimum relative lift → `form.optimizer.minRelativeLift`
   - Prior conviction (0–1) → `form.optimizer.priorConviction`
   - Metric std dev (continuous only) → `form.optimizer.metricStdDev`
   - Max experiment days → `form.optimizer.maxDays`

7. **[Calculate]** button — calls `POST /api/optimize` with optimizer fields; on success writes `form.mde`, `form.sampleSize`, `form.power`, `form.durationEstimate` from the result (so Config tab pre-fills). Displays GO / NO-GO result card below the button.

**Removed:** hypothesis, expected results, target population (free text), "Enter stats manually →" link.

---

## Tab 3 — Config

All values are written by the optimizer (via Calculate) but remain fully manually overridable.

**Fields:**
- Min. Detectable Effect → `form.mde`
- Sample size (per variant) → `form.sampleSize`
- Statistical power % → `form.power`
- Significance level (α) → `form.significanceLevel`
- Statistical test → `form.statisticalTest` (dropdown, existing `STAT_TESTS` store array)
- Duration estimate → `form.durationEstimate`
- Variants — name / description / traffic % grid; Add Variant and Remove buttons (minimum 2 variants enforced)

---

## Tab 4 — Instrumentation

**Fields:**
- Flag key → `form.flagKey`
- Event lineage → `form.eventLineage`

**Removed:** flagging system (defined in Global Settings), tracking tools (defined in Global Settings), metric computation.

---

## Data Model

Experiments are persisted to **SQLite** via the existing `/api/experiments` REST endpoints. The current frontend uses localStorage (key `exp_v2`) but the wizard redesign replaces that with API calls — localStorage is removed from the experiment flow entirely.

### Backend: new columns — `experiments` table

Appended to the `migrations` array in `run_migrations()` in `main.v` (errors on duplicate are swallowed with `or {}`):

```sql
ALTER TABLE experiments ADD COLUMN description TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN owner TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN primary_metric_slug TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN population_slugs TEXT DEFAULT '[]'
ALTER TABLE experiments ADD COLUMN guardrail_slugs TEXT DEFAULT '[]'
ALTER TABLE experiments ADD COLUMN learning_metric_slugs TEXT DEFAULT '[]'
ALTER TABLE experiments ADD COLUMN flag_key TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN event_lineage TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN variants TEXT DEFAULT '[]'
ALTER TABLE experiments ADD COLUMN mde TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN duration_estimate TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN statistical_test TEXT DEFAULT ''
ALTER TABLE experiments ADD COLUMN power REAL DEFAULT 80
ALTER TABLE experiments ADD COLUMN significance_level REAL DEFAULT 0.05
```

### Backend: `Experiment` struct additions (main.v lines 24–47)

```v
description          string
owner                string
primary_metric_slug  string
population_slugs     string
guardrail_slugs      string
learning_metric_slugs string
flag_key             string
event_lineage        string
variants             string
mde                  string
duration_estimate    string
statistical_test     string
power                f64
significance_level   f64
```

### Backend: `CreateExperimentReq` and `UpdateExperimentReq` additions

Same 14 fields as the struct additions above, added to both request structs.

### Backend: handler updates

- `create_experiment`: include new fields in the `Experiment{}` literal passed to `sql app.db { insert exp into Experiment }`. Remove the raw SQL UPDATE workaround for optimizer fields — instead add `optimizer_baseline`, `optimizer_daily_traffic`, `optimizer_min_relative_lift`, `optimizer_prior_conviction`, `optimizer_metric_std_dev`, `optimizer_worth_running` to the `Experiment` struct (they already exist as columns from prior migrations) and set them in the struct literal.
- `update_experiment`: extend the `sql app.db { update Experiment set ... }` statement with the new fields.
- `list_experiments` and `get_experiment`: no changes — V's ORM automatically includes new struct fields in SELECT.

### Frontend: form shape (app.js)

New fields in both the literal form initializer and the `$watch` reset block:

```js
description: '',
owner: '',
primaryMetricSlug: '',
populationSlugs: [],
guardrailSlugs: [],
learningMetricSlugs: [],
flagKey: '',
eventLineage: '',
variants: [
    { name: 'Control', description: '', traffic: 50 },
    { name: 'Treatment', description: '', traffic: 50 },
],
mde: '',
durationEstimate: '',
statisticalTest: 'Two-proportion z-test',
power: 80,
significanceLevel: 0.05,
```

Fields removed from the new form: `area`, `type`, `result`, `startDate`, `endDate`, `hypothesis`, `targetPopulation`, `mainMetrics`, `guardrails`, `expectedResults`, `flaggingSystem`, `trackingTools`, `metricComputation`, `events`.

### Frontend: store init — load from API

In `Alpine.store('app').init()`, replace:
```js
try { stored = JSON.parse(localStorage.getItem('exp_v2') || 'null'); } catch { ... }
```
with:
```js
const data = await fetch('/api/experiments').then(r => r.json()).catch(() => []);
this.experiments = data;
```

### Frontend: saveExp — call API

Replace the localStorage-based `saveExp(data)` with an async API call:
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
    if (!resp.ok) throw new Error('save failed');
    const saved = await resp.json();
    const idx = this.experiments.findIndex(x => x.id === saved.id);
    if (idx >= 0) this.experiments[idx] = saved;
    else this.experiments.unshift(saved);
    if (this.selectedExp?.id === saved.id) this.selectedExp = saved;
    this.modalExp = null;
    return saved;
},
```

### Frontend: deleteExp — call API

```js
async deleteExp(id) {
    if (!confirm('Delete this experiment? This cannot be undone.')) return;
    await fetch(`/api/experiments/${id}`, { method: 'DELETE' });
    this.experiments = this.experiments.filter(e => e.id !== id);
    if (this.selectedExp?.id === id) this.selectedExp = null;
},
```

---

## Frontend Architecture

### `expModal()` component changes

**Tab array:** `['General', 'Setup', 'Config', 'Instrumentation']` (replaces `['Basic','Summary','Design','Config','Instrumentation']`)

**New methods:**
- `async saveAndContinue()` — validates required fields for current tab, calls `this.app.saveExp(this.buildPayload())` (which calls POST if no id or PUT if id exists); on success advances to next tab
- `async saveAndClose()` — calls `this.app.saveExp(this.buildPayload())` and closes modal on success
- `buildPayload()` — constructs the API payload from `this.form`, mapping JS camelCase fields to snake_case keys expected by the backend (e.g. `primaryMetricSlug → primary_metric_slug`, `populationSlugs → population_slugs`, `variants → JSON.stringify(form.variants)`)
- `get id()` — returns `this.form.id` (set by server after first POST)
- `async refreshBaseline()` — POSTs to `/api/metrics/:selectedMetricId/refresh`, writes returned `value` to `form.optimizer.baseline`; disables button while running
- `togglePopulation(slug)` — toggles slug in `form.populationSlugs`
- `toggleGuardrail(slug)` — toggles slug in `form.guardrailSlugs`
- `toggleLearningMetric(slug)` — toggles slug in `form.learningMetricSlugs`
- `get selectedMetricId()` — finds metric id from `$store.app.metrics` by matching `primaryMetricSlug`
- `propagateOptimizerResult(result)` — called after Calculate; writes mde/sampleSize/power/durationEstimate from result into form

**Modified `runOptimizer()`:** After receiving the result, calls `propagateOptimizerResult(result)`.

**Removed:** `arrayToText()`, `textToArray()` (no longer needed — metrics/populations are catalog selects, not free-text textareas).

**The old single `save()` method is replaced by `saveAndContinue()` and `saveAndClose()`.**

### Modal footer HTML structure

```html
<div class="modal-footer">
  <button x-show="tab !== 'General'" @click="prevTab()">← Back</button>
  <div style="flex:1"></div>
  <button @click="$store.app.closeModal()">Close</button>
  <template x-if="tab !== 'Instrumentation'">
    <button @click="saveAndContinue()">Save & Continue →</button>
  </template>
  <template x-if="tab === 'Instrumentation'">
    <button @click="saveAndClose()">Save & Close</button>
  </template>
</div>
```

---

## Backend Changes

All in `apps/tracker/main.v`:

1. **14 `ALTER TABLE` migrations** — appended to the `migrations` array in `run_migrations()` (see Data Model above)
2. **`Experiment` struct** — 14 new fields added (lines 24–47 area)
3. **`CreateExperimentReq`** — 14 new fields; optimizer fields now set in struct literal (remove raw SQL UPDATE workaround)
4. **`UpdateExperimentReq`** — 14 new fields; optimizer fields added here too
5. **`create_experiment` handler** — populate new fields from req; remove the raw SQL UPDATE for optimizer fields
6. **`update_experiment` handler** — extend the `sql app.db { update Experiment set ... }` to include all new fields

The `POST /api/metrics/:id/refresh` endpoint (already implemented) handles the "Refresh now" button — no changes needed there.

---

## Sidebar panel (read-only impact)

The sidebar summary tab currently displays `targetPopulation`, `mainMetrics`, `guardrails`, `hypothesis`, `expectedResults`. After the redesign:

- `targetPopulation` display → replaced by population names resolved from `populationSlugs` against `$store.app.populations`
- `mainMetrics` display → replaced by metric name resolved from `primaryMetricSlug` against `$store.app.metrics`
- `guardrails` display → metric names from `guardrailSlugs`
- `learningMetricSlugs` → new section "Learning metrics" in summary tab
- `hypothesis` and `expectedResults` → removed from sidebar display

---

## Out of Scope

- Users/Owners CRUD page (sub-project 5 — owner remains a free-text field for now)
- Population and metric multi-select search/filter UI (list is short enough for checkboxes)
- Wizard step validation beyond name + owner on General tab
- Draft/autosave (Save & Continue is explicit)
- Sidebar edit of status/type/dates redesign
