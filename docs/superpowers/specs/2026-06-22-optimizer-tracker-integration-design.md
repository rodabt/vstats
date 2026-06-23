# Design: Optimizer-Integrated Experiment Creation

**Date:** 2026-06-22  
**Scope:** `apps/tracker` — experiment creation flow + backend API  
**Status:** Approved

---

## Problem

The tracker's experiment creation form asks users to manually supply `mde`, `sampleSize`, `power`, `significanceLevel`, and `durationEstimate`. These are statistical parameters most practitioners find difficult to reason about in isolation. The design optimizer (`vstats.experiment.optimizer_config` + `find_optimal_runtime`) can derive all of them from four plain-English inputs. Currently there is no connection between the two.

---

## Decision Summary

| Question | Decision | Rationale |
|---|---|---|
| Where does the optimizer sit? | Gated step (Option A) | Go/no-go is a gate, not a helper; checking feasibility before detailed planning avoids wasted work |
| Show detection rate chart? | Numbers only | Simpler modal, faster to scan; the chart is available in the optimizer example if needed |
| Store optimizer inputs? | Yes, as DB columns | Enables re-optimization when traffic changes; creates an audit trail for runtime decisions |
| Hard or soft gate on NO? | Soft gate with override | Teams sometimes must run experiments regardless; blocking causes tracker avoidance |

---

## User Flow

The creation modal gains a **Design** tab inserted between Summary and Config:

```
Basic → Summary → Design (new) → Config → Instrumentation
```

### Design tab

**Inputs (always shown):**

| Label | Field | Notes |
|---|---|---|
| Current metric value | `baseline` | Rate (0–1) for proportions; mean value for continuous |
| Daily users per variant | `daily_traffic_per_variant` | Integer |
| Minimum relative lift | `min_relative_lift` | e.g. 0.05 = detect ≥5% improvement |
| Prior conviction | `prior_conviction` | 0.0 = very skeptical · 1.0 = very confident |
| Metric std dev *(continuous only)* | `metric_std_dev` | Shown when metric_type = 'continuous'; leave 0 for proportions |
| Max experiment days | `max_days` | Defaults to 90 |

**"Calculate" button** calls `POST /api/optimize`. While loading, button shows a spinner.

**Result panel (numbers only):**

```
Worth running:        YES  /  NO — [reason]
Recommended runtime:  N days
Power at optimal:     N%
Monthly detection:    N.NNN
MDE (absolute):       N.NNpp  /  $N.NN
```

**If NO:** Amber warning box showing the `no_go_reason` string. User must tick "I understand this experiment is underpowered — proceed anyway" to enable the Next button. The override is stored as `optimizer_worth_running = 0` on the record.

**If YES:** Green confirmation panel. Next button enabled automatically.

**Manual escape hatch:** A small "Enter stats manually →" link skips the Design tab entirely and advances to Config with no pre-fill. This covers power users who receive MDEs from stakeholders.

### Pre-fill behaviour

On YES (or override), these Config fields are auto-populated and remain editable:

| Config field | Source |
|---|---|
| `mde` | `baseline × min_relative_lift` formatted as string |
| `sampleSize` | `optimal_days × daily_traffic_per_variant` |
| `power` | `power_at_optimal × 100` (rounded to nearest integer) |
| `significanceLevel` | `alpha` from config (0.05 default) |
| `durationEstimate` | `"${optimal_days} days"` |

---

## Backend

### New endpoint

```
POST /api/optimize
```

**Request body:**

```json
{
  "baseline": 0.41,
  "daily_traffic_per_variant": 1370,
  "min_relative_lift": 0.05,
  "prior_conviction": 0.30,
  "metric_std_dev": 0.0,
  "max_days": 90
}
```

**Response:**

```json
{
  "worth_running": true,
  "optimal_days": 14,
  "power_at_optimal": 0.982,
  "monthly_detection_rate": 0.584,
  "mde_absolute": 0.0205,
  "no_go_reason": "",
  "power_min_days": 7,
  "effective_min_days": 14
}
```

The handler calls `experiment.optimizer_config(params)` then `experiment.find_optimal_runtime(config)` directly — no reimplementation in JS. No result is persisted; `/api/optimize` is stateless.

### Schema additions

Six nullable columns added to the `experiments` table via a migration run at startup (using `IF NOT EXISTS` so it is safe to re-run):

```sql
ALTER TABLE experiments ADD COLUMN optimizer_baseline            REAL;
ALTER TABLE experiments ADD COLUMN optimizer_daily_traffic       INTEGER;
ALTER TABLE experiments ADD COLUMN optimizer_min_relative_lift   REAL;
ALTER TABLE experiments ADD COLUMN optimizer_prior_conviction    REAL;
ALTER TABLE experiments ADD COLUMN optimizer_metric_std_dev      REAL;
ALTER TABLE experiments ADD COLUMN optimizer_worth_running       INTEGER;  -- 0=no, 1=yes, NULL=not run
```

These are saved when the experiment is created (included in `CreateExperimentReq`) and updated when re-optimized. Existing records have all columns as NULL, meaning "optimizer not used."

---

## Component Boundaries

| Unit | Responsibility |
|---|---|
| `POST /api/optimize` handler | Decode request → call optimizer → encode response. No DB access. |
| Design tab (frontend) | Collect inputs, call `/api/optimize`, render result panel, manage override checkbox, pre-fill Config fields on advance |
| Config tab (frontend) | Unchanged except fields now arrive pre-filled; user can edit freely |
| DB migration (startup) | Add the six new columns idempotently |
| `CreateExperimentReq` / `UpdateExperimentReq` | Extended with six optimizer fields; all nullable |

---

## Out of Scope

- Re-optimize button on existing experiments (future work; schema is ready)
- Saving the full `OptimizationResult.all_results` curve (not needed for this feature)
- Changes to the readout or learnings flow
- Multi-variant optimizer support (current optimizer assumes two-arm tests)

---

## Open Questions (resolved)

All design questions were resolved during brainstorming. No open items.
