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

- **Save & Continue**: calls `this.app.saveExp(this.form)` (persists to localStorage), then advances to next tab
- **Close**: closes the modal; unsaved changes on the current tab are lost, but a prior Save & Continue has already persisted the form
- **Save & Close**: calls `saveExp` and closes the modal; equivalent to Save & Continue on the last tab

---

## Tab 1 — General

**Fields:**
- Name (text, required)
- Description (textarea, optional)
- Owner (text, required)

**Removed from this tab:** area/team (future Owners CRUD), status, type, result, start date, end date. These remain editable from the sidebar panel after creation. New experiments default to `status='planning'`, no dates.

**Save & Continue behavior:** Validates name and owner are non-empty, then calls `saveExp(form)` (localStorage). The form's `id` is generated on first save (`exp-${Date.now()}`) if not already set; all subsequent tabs reuse the same id.

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

Experiments are stored in **localStorage** (key `exp_v2`) via `Alpine.store('app').persist()`. The backend SQLite `experiments` table and API (`/api/experiments`) exist but are not used by the primary create/edit flow. No schema migration is required.

### Form shape additions (app.js)

```js
// new fields in form initializer (both in the literal object and the $watch reset)
primaryMetricSlug: '',
populationSlugs: [],
guardrailSlugs: [],
learningMetricSlugs: [],
```

These fields are included automatically in the `saveExp(data)` call (which spreads `this.form`) and written to localStorage alongside existing fields. When a saved experiment is reopened in the wizard, `init()` hydrates them from the stored object — no special parsing needed since localStorage stores them as JS values.

---

## Frontend Architecture

### `expModal()` component changes

**Tab array:** `['General', 'Setup', 'Config', 'Instrumentation']` (replaces `['Basic','Summary','Design','Config','Instrumentation']`)

**New methods:**
- `async saveAndContinue()` — validates required fields for current tab, calls POST (if `!form.id`) or PUT (if `form.id`), advances tab on success
- `async saveAndClose()` — like saveAndContinue but closes modal after save
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

None. Experiments are persisted entirely via localStorage — the backend experiment API is not involved in the wizard flow. The `POST /api/metrics/:id/refresh` endpoint (already implemented) is the only backend call the wizard makes, triggered by the "Refresh now" button.

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
