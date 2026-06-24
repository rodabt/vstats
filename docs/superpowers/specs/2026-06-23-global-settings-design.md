# Design: Global Settings Page

**Date:** 2026-06-23
**Scope:** `apps/tracker` — new `/settings` route + two managed lists + API endpoints
**Status:** Approved

---

## Problem

Flagging systems and tracking tools are currently hardcoded JavaScript constants in `app.js`. Each experiment stores its own selections, but in practice a team uses one flagging system and a fixed analytics stack. There is no place to manage these lists or add custom entries. The upcoming experiment creation wizard needs to source both lists from a database so teams can maintain their own options.

---

## Decision Summary

| Question | Decision | Rationale |
|---|---|---|
| Page vs panel/modal | Dedicated `/settings` page | Multiple editable lists outgrow a panel; a route is linkable and expandable |
| One list vs per-experiment selection | Global managed list, per-experiment pick from that list | Teams add to the canonical list once; experiments draw from it |
| Storage | SQLite (two new tables) | Consistent with all other tracker admin data |
| Pre-seed | Yes — seed from existing hardcoded values on first run | Zero config for existing users |

---

## Navigation

A gear icon `⚙` is added to the right side of the board header. Clicking it navigates to `/settings`. A `← Board` back link in the settings header returns to `/`.

The settings page has a simple two-section layout (no tabs needed at this scope):

```
Settings
─────────────────────────────────────────
Flagging Systems         [+ Add]
  • LaunchDarkly         [Edit] [Delete]
  • Statsig              [Edit] [Delete]
  • ...

Tracking Tools           [+ Add]
  • Amplitude            [Edit] [Delete]
  • Mixpanel             [Edit] [Delete]
  • ...
```

---

## Data Model

Two new SQLite tables, added via `run_migrations()` at startup:

```sql
CREATE TABLE IF NOT EXISTS flagging_systems (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tracking_tools (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);
```

**Pre-seed** (inserted once, skipped if rows already exist):

- Flagging systems: LaunchDarkly, Statsig, Optimizely, Split.io, GrowthBook, Custom / Internal
- Tracking tools: Amplitude, Mixpanel, Segment, BigQuery, Snowflake, Looker, dbt, Custom pipeline

The existing hardcoded `FLAGGING_SYSTEMS` and `TRACKING_TOOLS` constants in `app.js` are removed once the API endpoints are live.

---

## API Endpoints

All endpoints live in `apps/tracker/main.v`.

### Flagging Systems

| Method | Path | Action |
|---|---|---|
| GET | `/api/settings/flagging-systems` | Return all rows as JSON array |
| POST | `/api/settings/flagging-systems` | Create new entry `{ name: string }` |
| PUT | `/api/settings/flagging-systems/:id` | Rename entry `{ name: string }` |
| DELETE | `/api/settings/flagging-systems/:id` | Delete entry |

### Tracking Tools

Same four endpoints under `/api/settings/tracking-tools`.

**Response shape (GET):**
```json
[{ "id": 1, "name": "LaunchDarkly" }, ...]
```

**Error handling:** POST/PUT returns 400 if `name` is blank; DELETE returns 404 if id not found; UNIQUE constraint violations return 409.

---

## Frontend

### New route: `/settings`

The veb server adds a handler for `GET /settings` that serves the same `index.html` (the Alpine SPA). The SPA uses `window.location.pathname` to render the settings view instead of the board.

### Alpine store additions

```js
flaggingSystems: [],   // loaded from API on settings page init
trackingTools: [],     // loaded from API on settings page init
```

### Settings page component (`settingsPage` Alpine.data)

- On mount: fetch both lists from their respective GET endpoints
- Each list renders as: name label + Edit (inline rename) + Delete button
- `+ Add` opens an inline input at the bottom of the list; Submit calls POST
- Edit switches the row to an inline input; Save calls PUT; Cancel restores
- Delete calls DELETE with confirmation (`confirm()`)

### Connection to the experiment wizard (future tasks)

When the Instrumentation tab loads:
- Flagging system: `<select>` populated from `$store.app.flaggingSystems` + an "Add new…" option that POSTs inline and refreshes the store
- Tracking tools: checkbox group or multi-select populated from `$store.app.trackingTools` + same inline add

This wiring is implemented in the wizard task, not here. This task only delivers the settings page and the populated store arrays.

---

## Out of Scope

- Users / Owners CRUD (separate sub-project)
- Metrics catalog (separate sub-project)
- Population catalog (separate sub-project)
- Default experiment parameters (not requested)
- Reordering list entries
- Soft-delete / archive (hard delete only)
