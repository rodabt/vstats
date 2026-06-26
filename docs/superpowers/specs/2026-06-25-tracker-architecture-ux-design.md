# Experiment Tracker — Architecture, UX & Workflow Improvement Roadmap

**Date:** 2026-06-25
**Status:** Approved design — ready for implementation planning
**Scope:** `apps/tracker`

## Purpose

The tracker helps growth PMs, product, and data-science teams concentrate and
automate experimentation in one place: a quick glance at what's happening, and a
simple, fast way to create / sequence / orchestrate experiments. This document
assesses the current implementation against four goals and lays out a sequenced
plan to close the gaps.

The four goals:

1. **Reuse / "three clicks and done"** — reuse metrics, baselines, populations,
   owners; minimize friction to create a new experiment.
2. **TOML spec generation** — emit a complete, self-contained spec per experiment
   so humans and AI agents can operate on it (taxonomy, test-plan docs, readout
   config).
3. **Chat ("talk to your experiments")** — Claude-API-backed chat over experiment
   context.
4. **Professional, cohesive design system** — replace the current ad-hoc palette
   with an intentional, distinctive visual language.

## Current State (assessment)

**Backend** — `main.v` (~1370 lines, single file): V / veb + SQLite. Full CRUD
REST API for experiments, metrics, populations, owners, flagging systems,
tracking tools, learnings, and results. A `/api/optimize` endpoint performs power
analysis. Demo seed data (9 experiments, 12 metrics, 8 populations, 6 owners).

**Frontend** — `public/` (`index.html` + `app.js` ~52 KB + `app.css` ~47 KB):
Alpine.js SPA. Board (kanban by status) / timeline / list views, a 4-step
creation wizard (General → Setup → Config → Instrumentation), a detail sidebar,
and a live "tweaks" panel for accent / background / density customization.

**Reuse foundation already exists:** metrics, populations, and owners are
first-class catalogs with slugs; experiments reference them by slug
(`primary_metric_slug`, `population_slugs[]`, `guardrail_slugs[]`,
`learning_metric_slugs[]`). The data model supports reuse; the *workflow* does
not yet make reuse fast.

### Gap summary

| Goal | Current state | Gap |
|------|---------------|-----|
| Reuse / "3 clicks" | Slug-referenced catalogs exist. | No clone, no templates, no catalog-first creation path. Reuse is possible but not fast. |
| TOML spec export | None. | No schema, no endpoint, no export action. |
| Chat | None. Live key committed in `api_key.md`. | No Claude client, context builder, or chat UI. Committed secret is a security issue. |
| Design system | 7 CSS variables defined; **61 distinct hardcoded hex colors** across the three files. User-facing "tweaks" panel adds incoherence. | No real token system; no constrained palette. |

## Design Decisions (locked)

- **First workstream to implement:** Design System.
- **Tweaks panel:** Remove. One opinionated look.
- **TOML consumer:** Documentation / handoff **and** chat context — export-only
  for now (no round-trip import).
- **Aesthetic:** *Quiet analytical* — near-monochrome neutrals + one accent
  (`#4F46E5` indigo); color reserved strictly for experiment status & data viz.
- **Color mode:** Light-first; tokens structured so dark mode is a later drop-in.
- **Color rule:** Strict — status / data-viz only.
- **Typography:** Keep DM Sans / DM Mono (already fits the direction).

---

## Part 1 — Design System (detailed; implement first)

**Direction:** *Quiet analytical.* Distinctiveness comes from restraint and
typography, not from color volume.

### 1. Token layer (core fix)

Two-tier token system in `app.css`:

- **Primitives:** `--gray-50 … --gray-900`, `--accent` (`#4F46E5`) + hover, and a
  4-color status ramp, each with a tint (background) variant:
  - `--status-running` (teal), `--status-planning` (amber),
    `--status-success` (green), `--status-fail` (red), plus
    `--status-*-tint` for soft backgrounds.
- **Semantic tokens** (the only thing components reference):
  `--bg`, `--surface`, `--border`, `--text`, `--text-muted`, `--accent`,
  `--focus-ring`.

Components reference **only** semantic tokens. Dark mode later = swap one
`:root` (or `[data-theme="dark"]`) block; no component edits.

### 2. Color migration

Sweep `app.css`, `app.js`, `index.html`; map each of the 61 hardcoded colors to
the nearest token; delete one-offs.

**Acceptance check:** grepping for hex literals (`#[0-9A-Fa-f]{3,6}`) in the three
files returns **only** the token definitions in the `:root` block.

### 3. Remove the tweaks panel

Delete:

- The accent / background / density picker UI in `index.html`.
- The `tweaks` store object in `app.js`.
- The `postMessage` edit-mode plumbing (`__activate_edit_mode`,
  `__deactivate_edit_mode`, `__edit_mode_available`).
- `boardBg` inline styles (`:style` on `.app-root`).

Density survives as a single fixed default (compact), not a user toggle.

### 4. Strict color rule (enforced by convention)

Status colors appear **only** on: status pills, board-column headers, and chart
series. Buttons / links / nav use neutrals + the single accent. Document the rule
in a comment block at the top of `app.css` so it survives future edits.

### 5. Component normalization pass

While migrating, unify: button variants, card / sidebar surfaces, focus states,
and the status-pill component — all token-driven and consistent.

---

## Part 2 — Roadmap (plan-level)

### B · Reuse / "three clicks and done" (next after design system)

The catalog foundation exists; the gap is speed of reuse.

- **Clone-experiment:** "Duplicate" produces a pre-filled draft; change name +
  dates → save (~3 clicks).
- **Templates:** save any experiment as a reusable template; "New from template"
  picker on creation.
- **Catalog-first wizard:** surface existing metrics / populations as the default
  path (pick from list); "create new" is the secondary action — so reuse is the
  path of least resistance.

### C · TOML spec export (export-only; docs/handoff + chat context)

- Define a stable `experiment.toml` schema combining the experiment and its
  referenced metrics / populations / owners into one self-contained document:
  taxonomy, hypothesis, design (test type, MDE, power, sample size),
  instrumentation (events, lineage), and readout config.
- V endpoint: `GET /api/experiments/:id/spec.toml`.
- "Copy / Download TOML" action in the detail sidebar.
- Export-only for now (no round-trip import).

### D · Chat — "talk to your experiments" (last; leans on C)

- **Security fix first (independent of sequencing):** `api_key.md` with a live
  key is committed to the repo. Move it to an env var / gitignored file and
  rotate the key before any chat work.
- V-side Claude API client + a context builder that feeds the relevant
  experiment's TOML spec (and a portfolio summary) as context.
- Chat UI panel, scoped to either one experiment or the whole portfolio.

### Sequence

**Design System → Reuse UX → TOML Spec → Chat.**
Design system leads (unblocks all visual work, lowest risk). Chat trails (depends
on the TOML context contract). The security fix for `api_key.md` happens before
any chat work regardless of overall timing.

## Out of Scope

- TOML round-trip import / agent-authored experiments (export-only for now).
- Dark theme implementation (tokens prepare for it; the theme itself is later).
- Backend refactor of `main.v` beyond what the TOML and chat endpoints require.
- Any new statistical / optimizer functionality.
