# Design: Population Catalog

**Date:** 2026-06-24
**Scope:** `apps/tracker` — new `/populations` route, SQLite `populations` table, 5 API endpoints, vduckdb-powered count refresh
**Status:** Approved

---

## Problem

Eligible populations for experiments are currently free-text on the experiment form (`targetPopulation`). There is no shared catalog, no stored count, and no way to compute the eligible user count automatically. The upcoming experiment wizard needs a multi-select dropdown populated from a managed list of populations, each with a SQL query that can fetch the current count from the data warehouse on demand.

---

## Decision Summary

| Question | Decision | Rationale |
|---|---|---|
| Edit UX | Modal for create/edit, inline Refresh/Delete | SQL textarea needs space; list stays scannable |
| Count execution | vduckdb in-memory, split SQL on `;` | Same pattern as metrics catalog (`run_metric_sql`) |
| Count storage | REAL in SQLite, returned as TEXT string | Consistent with `baseline_value` in metrics; flexible for approximate counts |
| Slug mutability | Auto-generated on create, read-only after first save | Matches metrics catalog pattern |

---

## Data Model

New SQLite table added via `run_migrations()` at startup:

```sql
CREATE TABLE IF NOT EXISTS populations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    slug                TEXT NOT NULL UNIQUE,
    sql_query           TEXT NOT NULL DEFAULT '',
    population_count    REAL,
    count_updated_at    TEXT
);
```

No `metric_type` — populations have no dimension axis. `population_count` is stored as REAL (same pattern as `baseline_value` in metrics) and formatted as an integer in the UI (e.g. `12,450`).

`slug` is auto-generated from `name` on create (lowercase, non-alphanumeric → underscores), editable before first save, then read-only.

`population_count` and `count_updated_at` are written by the refresh endpoint and persisted so the wizard can display the last known count without re-running SQL.

---

## API Endpoints

All endpoints live in `apps/tracker/main.v`.

| Method | Path | Action |
|---|---|---|
| GET | `/api/populations` | Return all rows ordered by name |
| POST | `/api/populations` | Create population `{ name, slug, sql_query }` |
| PUT | `/api/populations/:id` | Update population (name, sql_query — slug immutable) |
| DELETE | `/api/populations/:id` | Delete population |
| POST | `/api/populations/:id/refresh` | Run SQL, store result, return `{ count, updated_at }` |

**GET response shape:**
```json
[{
  "id": "1",
  "name": "Mobile users — last 30d",
  "slug": "mobile_users_30d",
  "sql_query": "INSTALL httpfs;\nLOAD httpfs;\n...",
  "population_count": "84320",
  "count_updated_at": "2026-06-24 10:32:00"
}]
```

**Error handling:**
- POST/PUT: 400 if `name` or `slug` is blank; 409 on UNIQUE violation (pre-insert SELECT check)
- DELETE: 200 always (idempotent)
- Refresh: 422 `{ "error": "<duckdb error message>" }` on SQL failure; 200 `{ "count": "84320", "updated_at": "..." }` on success

---

## Count Execution

`POST /api/populations/:id/refresh` reuses the existing `run_metric_sql` helper:

1. Fetches the population's `sql_query` from SQLite
2. Calls `run_metric_sql(sql_query)` → opens in-memory vduckdb, splits on `;`, runs each statement, reads first column of `get_first_row()`
3. Updates `population_count` and `count_updated_at` in SQLite
4. Returns `{ "count": "<scalar>", "updated_at": "<timestamp>" }`

No new DuckDB helper needed — `run_metric_sql` is reused as-is.

---

## Navigation

A "users/people" SVG icon is added to the board header, to the left of the BarChart (metrics) icon. Clicking it navigates to `/populations`. Pattern mirrors `/settings` and `/metrics`:

- `GET /populations` in `main.v` → serves `index.html`
- Store getter: `get isPopulations() { return window.location.pathname === '/populations'; }`
- `loadPopulations()` called in store `init()`
- Board `x-if` guard extended: `!$store.app.isSettings && !$store.app.isMetrics && !$store.app.isPopulations`

---

## Frontend

### Alpine store additions

```js
populations: [],
get isPopulations() { return window.location.pathname === '/populations'; },
async loadPopulations() {
    const data = await fetch('/api/populations').then(r => r.json());
    this.populations = data;
},
```

`loadPopulations()` called at the end of store `init()`.

### `populationsPage` Alpine.data component

```js
Alpine.data('populationsPage', () => ({
    get populations() { return Alpine.store('app').populations; },
    showModal: false,
    editing: null,
    form: { name: '', slug: '', sql_query: '' },
    refreshing: {},

    slugify(name) { ... },
    onNameInput() { ... },
    openNew() { ... },
    openEdit(pop) { ... },
    closeModal() { ... },
    async save() { ... },
    async deletePop(id, name) { ... },
    async refresh(id) { ... },
    formatCount(count) { ... },   // formats "84320" → "84,320"
    formatAge(updated_at) { ... },
}))
```

No `metric_type` field — the modal has name, slug, and SQL query only.

### Page layout

```
← Board

Populations                                       [+ New population]
────────────────────────────────────────────────────────────────────
Mobile users — last 30d    84,320   2h ago    [↻] [Edit] [Delete]
Checkout page visitors      12,450  1d ago    [↻] [Edit] [Delete]
All active users                —   never     [↻] [Edit] [Delete]
```

The `↻` refresh button is per-row; while refreshing it shows a spinner and is disabled. On success the count and timestamp update in place.

### Modal

- **Name** — text input; on create, typing auto-fills Slug
- **Slug** — text input (editable on create, `disabled` on edit)
- **SQL query** — `<textarea rows="12">` monospace font; placeholder shows minimal DuckDB example returning a count

---

## CSS

New classes appended to `app.css` following `.metrics-*` naming convention:
`.populations-page`, `.populations-back`, `.populations-header`, `.populations-title`, `.populations-add-btn`, `.populations-list`, `.populations-row`, `.populations-row-name`, `.populations-row-count`, `.populations-row-age`, `.populations-row-actions`, `.populations-btn-refresh`, `.populations-btn-edit`, `.populations-btn-delete`, `.populations-empty`, `.populations-modal-overlay`, `.populations-modal-box`, `.populations-modal-title`, `.populations-modal-field`, `.populations-modal-label`, `.populations-modal-input`, `.populations-modal-select`, `.populations-modal-textarea`, `.populations-modal-footer`, `.populations-btn-cancel`, `.populations-btn-save`

---

## Connection to the Experiment Wizard (future task)

When the wizard's Setup tab loads:
- Eligible populations: multi-select or checkbox group populated from `$store.app.populations`
- Selecting populations stores their slugs in the experiment record
- Count displayed alongside each option helps the experimenter estimate traffic split

This wiring is implemented in the wizard task, not here. This task only delivers the catalog page and the populated store array.

---

## Out of Scope

- Experiment wizard redesign (separate sub-project)
- Population versioning or history
- Population-level access control
- SQL syntax validation (DuckDB error surfaced as-is)
- Population ordering / drag-to-reorder
- Combining multiple populations (union logic belongs in wizard)
