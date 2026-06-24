# Design: Metrics Catalog

**Date:** 2026-06-24
**Scope:** `apps/tracker` — new `/metrics` route, SQLite `metrics` table, 5 API endpoints, vduckdb-powered baseline refresh
**Status:** Approved

---

## Problem

Experiment metrics are currently free-text arrays on each experiment form. There is no shared catalog, no stored baseline values, and no way to compute a baseline automatically. The upcoming experiment wizard needs a dropdown populated from a managed list of metrics, each with a known type (proportion vs. continuous) and a SQL query that can fetch the current baseline from the data warehouse on demand.

---

## Decision Summary

| Question | Decision | Rationale |
|---|---|---|
| Edit UX | Modal for create/edit, inline Refresh/Delete | SQL textarea needs space; list stays scannable |
| Baseline execution | vduckdb in-memory, split SQL on `;` | Matches cuiqData/cuiqGEN patterns; no subprocess overhead |
| Baseline storage | Yes — stored after each refresh | Wizard reads stored value without re-running SQL |
| Slug mutability | Auto-generated on create, read-only after first save | Avoids breaking experiment references |
| metric_type values | `'proportion'` \| `'continuous'` | Needed by the optimizer to determine std dev treatment |

---

## Data Model

New SQLite table added via `run_migrations()` at startup:

```sql
CREATE TABLE IF NOT EXISTS metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    slug                TEXT NOT NULL UNIQUE,
    metric_type         TEXT NOT NULL DEFAULT 'proportion',
    sql_query           TEXT NOT NULL DEFAULT '',
    baseline_value      REAL,
    baseline_updated_at TEXT
);
```

`metric_type` is `'proportion'` (rate 0–1, e.g. CVR, CTR) or `'continuous'` (unbounded, e.g. revenue, latency).

`slug` is auto-generated from `name` on create (lowercase, spaces and special chars → underscores), editable in the modal before first save, then read-only.

`baseline_value` and `baseline_updated_at` are written by the refresh endpoint and persisted so the wizard can display the last known value without re-running SQL.

---

## API Endpoints

All endpoints live in `apps/tracker/main.v`.

| Method | Path | Action |
|---|---|---|
| GET | `/api/metrics` | Return all rows ordered by name |
| POST | `/api/metrics` | Create metric `{ name, slug, metric_type, sql_query }` |
| PUT | `/api/metrics/:id` | Update metric (name, metric_type, sql_query — slug is immutable) |
| DELETE | `/api/metrics/:id` | Delete metric |
| POST | `/api/metrics/:id/refresh` | Run SQL, store result, return `{ value, updated_at }` |

**GET response shape:**
```json
[{
  "id": "1",
  "name": "Checkout CVR",
  "slug": "checkout_cvr",
  "metric_type": "proportion",
  "sql_query": "INSTALL httpfs;\nLOAD httpfs;\n...",
  "baseline_value": "0.041",
  "baseline_updated_at": "2026-06-24 10:32:00"
}]
```

**Error handling:**
- POST/PUT: 400 if `name` or `slug` is blank; 409 on UNIQUE violation (checked via pre-insert SELECT)
- DELETE: 200 always (idempotent)
- Refresh: 422 with `{ "error": "<duckdb error message>" }` if SQL fails; 200 with `{ "value": "0.041", "updated_at": "..." }` on success

---

## Baseline Execution

The `POST /api/metrics/:id/refresh` handler:

1. Fetches the metric's `sql_query` from SQLite
2. Opens an in-memory vduckdb connection (`vdb.open(':memory:')`)
3. Splits the SQL on `;`, trims whitespace, filters empty strings
4. Runs each statement with `vdb.query(stmt)` — stops on first error, returns 422
5. Calls `vdb.get_first_row()` on the final statement's result
6. Takes the first column value as the scalar baseline
7. Updates `baseline_value` and `baseline_updated_at` in SQLite
8. Returns `{ "value": "<scalar>", "updated_at": "<timestamp>" }`

Helper function signature:
```v
fn run_metric_sql(sql_query string) !(string, string)
// returns (value_string, updated_at_string) or error
```

**v.mod:** Add `{ name: 'vduckdb' }` to `apps/tracker/v.mod` dependencies.

---

## Navigation

A "metrics" SVG icon is added to the board header, to the left of the existing gear icon. Clicking it navigates to `/metrics`. The pattern mirrors `/settings`:

- `GET /metrics` in `main.v` → serves `index.html`
- Store getter: `get isMetrics() { return window.location.pathname === '/metrics'; }`
- `loadMetrics()` called in store `init()`
- Board `x-if` guard extended: `!$store.app.isSettings && !$store.app.isMetrics`

---

## Frontend

### Alpine store additions

```js
metrics: [],
get isMetrics() { return window.location.pathname === '/metrics'; },
async loadMetrics() {
    const data = await fetch('/api/metrics').then(r => r.json());
    this.metrics = data;
},
```

`loadMetrics()` called at the end of store `init()`.

### `metricsPage` Alpine.data component

```js
Alpine.data('metricsPage', () => ({
    get metrics() { return Alpine.store('app').metrics; },
    showModal: false,
    editing: null,       // null = new metric, number = id being edited
    form: { name: '', slug: '', metric_type: 'proportion', sql_query: '' },
    refreshing: {},      // { [id]: true } while a refresh is in flight

    openNew() { ... },
    openEdit(metric) { ... },
    closeModal() { ... },
    save() { ... },      // POST /api/metrics or PUT /api/metrics/:id
    deleteMetric(id, name) { ... },
    async refresh(id) { ... },  // POST /api/metrics/:id/refresh
}))
```

Slug auto-populates from name on `openNew()` via an `x-model` watcher. On `openEdit()` the slug field is disabled (read-only).

### Page layout

```
← Board

Metrics Catalog                                     [+ New metric]
────────────────────────────────────────────────────────────────
Checkout CVR         proportion   0.041   2h ago   [↻] [Edit] [Delete]
Revenue per session  continuous  47.30    1d ago   [↻] [Edit] [Delete]
D7 activation rate   proportion    —      never    [↻] [Edit] [Delete]
```

The `↻` (refresh) button is per-row; while refreshing, it shows a spinner and is disabled. On success, the baseline value and timestamp update in place (store reloaded).

### Modal

The create/edit modal contains:
- **Name** — text input; on create, typing here auto-fills Slug
- **Slug** — text input (editable on create, `disabled` attribute on edit)
- **Type** — `<select>`: `Proportion (rate 0–1)` / `Continuous (e.g. revenue)`
- **SQL query** — `<textarea rows="12">` with monospace font; placeholder shows a minimal DuckDB multi-statement example

---

## CSS

New classes appended to `app.css` following the same naming convention as `.settings-*`:
`.metrics-page`, `.metrics-list`, `.metrics-row`, `.metrics-row-name`, `.metrics-row-type`, `.metrics-row-baseline`, `.metrics-row-age`, `.metrics-row-actions`, `.metrics-btn-refresh`, `.metrics-btn-edit`, `.metrics-btn-delete`, `.metrics-modal`, `.metrics-modal-overlay`, `.metrics-modal-box`, `.metrics-modal-title`, `.metrics-modal-field`, `.metrics-modal-label`, `.metrics-modal-input`, `.metrics-modal-select`, `.metrics-modal-textarea`, `.metrics-modal-footer`, `.metrics-btn-save`, `.metrics-btn-cancel`, `.metrics-add-btn`

---

## Connection to the Experiment Wizard (future task)

When the wizard's Setup tab loads:
- Primary metric: `<select>` populated from `$store.app.metrics`
- Selecting a metric sets `form.optimizer.baseline` from `metric.baseline_value` and `form.metric_type`
- "Refresh now" button calls `POST /api/metrics/:id/refresh` and updates `form.optimizer.baseline` with the returned value

This wiring is implemented in the wizard task, not here. This task only delivers the catalog page and the populated store array.

---

## Out of Scope

- Population catalog (separate sub-project)
- Experiment wizard redesign (separate sub-project)
- Metric versioning or history
- Metric-level access control
- SQL syntax validation (DuckDB error is surfaced as-is)
- Metric ordering / drag-to-reorder
