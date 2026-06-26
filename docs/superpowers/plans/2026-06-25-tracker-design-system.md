# Tracker Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tracker's 60 ad-hoc hex colors with a two-tier design-token system in the *Quiet analytical* direction, remove the per-user "tweaks" panel, and enforce a strict status-only color rule — so the UI reads as one intentional, professional product.

**Architecture:** A single `:root` token block in `public/app.css` defines primitives (neutral ramp + accent + status hues) and semantic tokens (`--bg`, `--surface`, `--border`, `--text`, `--accent`, status lifecycle colors). Every component — CSS rules, the three JS color-config objects, and inline `:style` bindings — references only semantic tokens. Light-first; the token structure makes dark mode a later one-block drop-in. No build step: plain CSS custom properties + Alpine.js.

**Tech Stack:** Plain CSS custom properties, Alpine.js 3, vanilla JS. No framework, no bundler. Backend (`main.v`, veb) is untouched by this plan.

## Global Constraints

- **Aesthetic:** Quiet analytical — near-monochrome neutrals + ONE accent (`#4F46E5` indigo). Verbatim from spec.
- **Color rule (strict):** Color appears ONLY on experiment status (lifecycle pills, board-column dots/headers, timeline bars), result/outcome badges, and data-viz. Everything else — buttons, links, nav, type badges — is neutral + the single accent. Verbatim from spec.
- **Color mode:** Light-first. Tokens structured so a future `[data-theme="dark"]` block is the only change needed. Verbatim from spec.
- **Typography:** Keep DM Sans / DM Mono. Do not change fonts.
- **Acceptance invariant:** After the full migration, grepping `public/app.css`, `public/app.js`, `public/index.html` for hex literals (`#[0-9A-Fa-f]{3,6}`) returns matches ONLY inside the `:root` token block in `app.css`. No hex literals anywhere else.
- **Server startup protocol (from project memory — REQUIRED before every `v run .`):**
  ```bash
  pkill -f apps/tracker/tracker 2>/dev/null; fuser -k 8080/tcp 2>/dev/null; sleep 1
  ```
  Multiple tracker binaries share port 8080 via SO_REUSEPORT; stale binaries stay resident. `pkill` is required to clear them or ~75% of requests hit old code.
- **Working directory:** `/home/rabt/devel/vstats/apps/tracker`. `apps/` is gitignored in the vstats repo, so the app's own files are NOT version-controlled there. Commit steps below cover only this plan/spec under `docs/` (force-added). The frontend file edits are real but uncommitted-to-git by design; "commit" steps for them are recorded in the plan as checkpoints but there is no git history for `apps/` — instead, each task's checkpoint is the **verification gate passing**, not a git commit. Where a step says "Commit", run it only for files under `docs/`; for `public/*` edits the gate IS the deliverable.

---

## Token Reference (used by every task)

This is the authoritative token block. Task 1 writes it verbatim into `public/app.css`, replacing the existing 7-variable `:root`.

```css
/* ============================================================
   DESIGN TOKENS — Quiet analytical
   COLOR RULE (strict): color appears ONLY on experiment status,
   result/outcome badges, and data-viz. Buttons, links, nav, and
   type badges use neutrals + the single accent. Do NOT introduce
   raw hex outside this block — add or reuse a token instead.
   Dark mode later = add a [data-theme="dark"] block overriding
   the SEMANTIC tokens only.
   ============================================================ */
:root {
	/* --- Primitives: neutral ramp --- */
	--gray-0:   #FFFFFF;
	--gray-50:  #FAFAFA;
	--gray-100: #F3F4F6;
	--gray-200: #E5E7EB;
	--gray-300: #D1D5DB;
	--gray-400: #9CA3AF;
	--gray-500: #6B7280;
	--gray-600: #4B5563;
	--gray-700: #374151;
	--gray-800: #1F2937;
	--gray-900: #16181D;

	/* --- Primitives: accent (single) --- */
	--indigo:       #4F46E5;
	--indigo-hover: #4338CA;
	--indigo-tint:  #EEF2FF;

	/* --- Primitives: status hues (+ soft tint for backgrounds) --- */
	--hue-slate:  #5B6472;  --hue-slate-tint:  #F1F3F5;
	--hue-amber:  #D97706;  --hue-amber-tint:  #FEF6EC;
	--hue-teal:   #0E9CA8;  --hue-teal-tint:   #ECFBFC;
	--hue-green:  #059669;  --hue-green-tint:  #ECFDF5;
	--hue-red:    #DC2626;  --hue-red-tint:    #FCEEEE;

	/* --- Semantic: surfaces & text --- */
	--bg:             var(--gray-50);
	--surface:        var(--gray-0);
	--surface-sunken: var(--gray-100);
	--border:         var(--gray-200);
	--border-strong:  var(--gray-300);
	--text:           var(--gray-900);
	--text-secondary: var(--gray-700);
	--text-muted:     var(--gray-500);
	--text-faint:     var(--gray-400);

	/* --- Semantic: accent --- */
	--accent:       var(--indigo);
	--accent-hover: var(--indigo-hover);
	--accent-tint:  var(--indigo-tint);
	--focus-ring:   rgba(79, 70, 229, 0.35);

	/* --- Semantic: status lifecycle --- */
	--status-planning:      var(--hue-slate);  --status-planning-tint:      var(--hue-slate-tint);
	--status-instrumenting: var(--hue-amber);  --status-instrumenting-tint: var(--hue-amber-tint);
	--status-running:       var(--hue-teal);   --status-running-tint:       var(--hue-teal-tint);
	--status-completed:     var(--hue-green);  --status-completed-tint:     var(--hue-green-tint);
	--status-cancelled:     var(--hue-red);    --status-cancelled-tint:     var(--hue-red-tint);
	--status-archived:      var(--gray-400);   --status-archived-tint:      var(--gray-100);

	/* --- Semantic: result/feedback --- */
	--success: var(--hue-green); --success-tint: var(--hue-green-tint);
	--warning: var(--hue-amber); --warning-tint: var(--hue-amber-tint);
	--danger:  var(--hue-red);   --danger-tint:  var(--hue-red-tint);
}
```

### Color → token mapping table (authoritative)

Apply to EVERY hex occurrence outside the `:root` block. Status/type/result hexes
inside the three JS config objects are handled centrally in Task 2; the rows below
cover the remaining CSS/HTML occurrences.

| Raw hex (normalized) | Token |
|---|---|
| `#FFFFFF` / `#FFF` | `var(--surface)` |
| `#FAFAFA`, `#F5F6F8`, `#F9FAFB`, `#F8FAFC` | `var(--bg)` |
| `#F3F4F6` | `var(--surface-sunken)` |
| `#E5E7EB` | `var(--border)` |
| `#D1D5DB` | `var(--border-strong)` |
| `#9CA3AF` | `var(--text-faint)` |
| `#6B7280` | `var(--text-muted)` |
| `#374151`, `#4B5563` | `var(--text-secondary)` |
| `#1F2937`, `#111827` | `var(--text)` |
| `#4F46E5`, `#2563EB` (non-status) | `var(--accent)` |
| `#4338CA`, `#1D4ED8` | `var(--accent-hover)` |
| `#EEF2FF`, `#EFF6FF`, `#F0F7FF`, `#DBEAFE`, `#BFDBFE`, `#F0F4FF`, `#3B82F6` (non-status) | `var(--accent-tint)` |
| `#059669`, `#10B981`, `#15803D`, `#166534`, `#065F46`, `#86EFAC` | `var(--success)` |
| `#ECFDF5`, `#F0FDF4` | `var(--success-tint)` |
| `#D97706`, `#F59E0B`, `#B45309`, `#92400E`, `#78350F`, `#FCD34D`, `#FDE68A` | `var(--warning)` |
| `#FFFBEB`, `#FEF3C7` | `var(--warning-tint)` |
| `#DC2626`, `#EF4444`, `#B91C1C`, `#7F1D1D`, `#FCA5A5`, `#F87171` | `var(--danger)` |
| `#FEF2F2`, `#FEE2E2` | `var(--danger-tint)` |
| `#7C3AED`, `#8B5CF6` (type/violet, non-status) | `var(--text-secondary)` (neutralized) |
| `#F5F3FF`, `#ECFEFF`, `#F0F9FF`, `#FDF2F8`, `#F0F9FF`, `#ECFEFF` | `var(--surface-sunken)` (neutralized type bg) |
| `#BE185D`, `#0891B2`, `#0EA5E9`, `#0369A1` (type/cyan, non-status) | `var(--text-secondary)` (neutralized) |
| `#FAFAF8` (tweaks "Warm" bg) | n/a — removed with tweaks panel (Task 5) |

> Disambiguation rule: a hex used as an experiment **status** color belongs to a
> `STATUS_CONFIG` entry and is migrated in Task 2. The same hex appearing in a
> button/nav/border context maps to the neutral/accent token per the table. When
> in doubt, neutral wins (strict color rule).

---

## Task 1: Token foundation

Add the full token block; the rest of the file still uses raw hex (migrated in
later tasks). Goal: tokens exist and the app still renders identically.

**Files:**
- Modify: `public/app.css:1-9` (replace the existing `:root` block)

**Interfaces:**
- Produces: all CSS custom properties listed in the Token Reference above. Every
  later task consumes these names.

- [ ] **Step 1: Replace the `:root` block**

In `public/app.css`, replace lines 1–9 (the current `:root { --accent … --success … }`
block) with the **entire token block** from the Token Reference section above
(the `/* DESIGN TOKENS */` comment through the closing `}`).

- [ ] **Step 2: Start the server and load the app**

```bash
pkill -f apps/tracker/tracker 2>/dev/null; fuser -k 8080/tcp 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run . 
```
Run the build in the background (so it stays up). Then load `http://localhost:8080`.
Expected: board renders exactly as before (no visual change yet — nothing
references the new tokens). No console errors.

- [ ] **Step 3: Verify tokens are defined**

```bash
grep -c "^\s*--" public/app.css | head -1
```
Expected: ≥ 45 (the token block defines ~50 custom properties).

- [ ] **Step 4: Commit (docs only — see Global Constraints)**

No git commit for `public/` (gitignored). Checkpoint = Steps 2–3 pass.

---

## Task 2: Migrate the JS color-config objects

`STATUS_CONFIG`, `TYPE_CONFIG`, and `RESULT_CONFIG` (top of `public/app.js`) drive
every status/type/result color via inline `:style`. Point them at tokens. Types
become neutral (strict rule: type is taxonomy, not status). Fix the one hex-alpha
concatenation in the timeline bar.

**Files:**
- Modify: `public/app.js:3-22` (the three config objects)
- Modify: `public/index.html:316` (hex-alpha concat) and `:329` (type-badge fallback)

**Interfaces:**
- Consumes: status/accent/feedback tokens from Task 1.
- Produces: `STATUS_CONFIG[status].{color,bg,dot}`, `TYPE_CONFIG[type].{color,bg}`,
  `RESULT_CONFIG[result].{label,color,bg}` — all now `var(--token)` strings.

- [ ] **Step 1: Replace `STATUS_CONFIG` (app.js:3-10)**

```javascript
const STATUS_CONFIG = {
	planning:      { label:'Planning',      color:'var(--status-planning)',      bg:'var(--status-planning-tint)',      dot:'var(--status-planning)' },
	instrumenting: { label:'Instrumenting', color:'var(--status-instrumenting)', bg:'var(--status-instrumenting-tint)', dot:'var(--status-instrumenting)' },
	running:       { label:'Running',       color:'var(--status-running)',       bg:'var(--status-running-tint)',       dot:'var(--status-running)' },
	completed:     { label:'Completed',     color:'var(--status-completed)',     bg:'var(--status-completed-tint)',     dot:'var(--status-completed)' },
	cancelled:     { label:'Cancelled',     color:'var(--status-cancelled)',     bg:'var(--status-cancelled-tint)',     dot:'var(--status-cancelled)' },
	archived:      { label:'Archived',      color:'var(--status-archived)',      bg:'var(--status-archived-tint)',      dot:'var(--status-archived)' },
};
```

- [ ] **Step 2: Replace `TYPE_CONFIG` (app.js:11-17) — neutralized**

```javascript
const TYPE_CONFIG = {
	'A/B':          { color:'var(--text-secondary)', bg:'var(--surface-sunken)' },
	'Multivariate': { color:'var(--text-secondary)', bg:'var(--surface-sunken)' },
	'Sequential':   { color:'var(--text-secondary)', bg:'var(--surface-sunken)' },
	'Holdout':      { color:'var(--text-secondary)', bg:'var(--surface-sunken)' },
	'Bandit':       { color:'var(--text-secondary)', bg:'var(--surface-sunken)' },
};
```

- [ ] **Step 3: Replace `RESULT_CONFIG` (app.js:18-22)**

```javascript
const RESULT_CONFIG = {
	success:      { label:'Success',      color:'var(--success)', bg:'var(--success-tint)' },
	inconclusive: { label:'Inconclusive', color:'var(--warning)', bg:'var(--warning-tint)' },
	fail:         { label:'Failed',       color:'var(--danger)',  bg:'var(--danger-tint)' },
};
```

- [ ] **Step 4: Fix the hex-alpha concat (index.html:316)**

`var()` values cannot have a hex-alpha suffix appended. Replace:

```html
									border:1.5px solid ${$store.app.STATUS_CONFIG[exp.status].dot}33;
```
with:
```html
									border:1.5px solid color-mix(in srgb, ${$store.app.STATUS_CONFIG[exp.status].dot} 22%, transparent);
```

- [ ] **Step 5: Fix the type-badge fallback hex (index.html:329)**

Replace `|| '#374151'` with `|| 'var(--text-secondary)'` in the `tl-bar` type badge `:style`:
```html
										<span class="badge xs" :style="'background:' + $store.app.TYPE_CONFIG[exp.test_type]?.bg + ';color:' + ($store.app.TYPE_CONFIG[exp.test_type]?.color || 'var(--text-secondary)')" x-text="exp.test_type"></span>
```

- [ ] **Step 6: Restart server and verify status colors**

```bash
pkill -f apps/tracker/tracker 2>/dev/null; fuser -k 8080/tcp 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run .
```
Load `http://localhost:8080`. Switch to **Board** and **Timeline** views.
Expected: status dots/pills show the new hues — planning=slate, instrumenting=amber,
running=teal, completed=green, cancelled=red, archived=gray. Type badges are now
neutral gray (no longer colored). Timeline bars render with a faint colored border
(no broken `…33` borders). No console errors.

- [ ] **Step 7: Grep gate for the configs**

```bash
grep -nE "#[0-9A-Fa-f]{3,6}" public/app.js | grep -E "STATUS_CONFIG|TYPE_CONFIG|RESULT_CONFIG"
```
Expected: no output (no hex remains in the three configs).

---

## Task 3: Migrate `app.css` to tokens

Apply the mapping table to every hex in `public/app.css` outside the `:root` block.
This is the bulk of the work (the file holds ~45 of the ~60 colors).

**Files:**
- Modify: `public/app.css` (all rules below `:root`)

**Interfaces:**
- Consumes: all semantic tokens from Task 1.

- [ ] **Step 1: Replace every hex occurrence using the mapping table**

Work top-to-bottom through `public/app.css`. For each hex literal below the
`:root` block, replace it with the token from the mapping table. Notes:
- Animation/keyframe colors (e.g. `#F87171` in `.field-error`) → `var(--danger)`.
- Scrollbar (`#D1D5DB`, `#9CA3AF`) → `var(--border-strong)`, `var(--text-faint)`.
- `body { background:#F5F6F8 }` → `background: var(--bg)`.
- Any status/result pill classes (`.badge.success`, `.status-*`) → the matching
  `--status-*` / `--success|warning|danger` token, NOT a neutral.
- Buttons, links, nav, tabs, inputs → neutrals + `var(--accent)` only.

- [ ] **Step 2: Grep gate — app.css clean except `:root`**

```bash
awk '/^:root/{r=1} /^}/{if(r){r=0; next}} !r' public/app.css | grep -nE "#[0-9A-Fa-f]{3,6}"
```
Expected: no output. (The `awk` strips the `:root` block; any remaining hex is a miss.)

- [ ] **Step 3: Restart server and visual smoke test**

```bash
pkill -f apps/tracker/tracker 2>/dev/null; fuser -k 8080/tcp 2>/dev/null; sleep 1
cd /home/rabt/devel/vstats/apps/tracker && v run .
```
Load the app. Click through Board, Timeline, List, open an experiment's detail
sidebar, open the creation wizard. Expected: cohesive neutral surfaces, single
indigo accent on buttons/links, status color only on status/result elements.
No unstyled/black-default elements, no console errors.

---

## Task 4: Migrate remaining inline hexes in `index.html` & `app.js`

Mop up hex literals outside the three JS configs (Task 2) and `app.css` (Task 3):
inline `:style` defaults, icon fills, and any seed-data colors.

**Files:**
- Modify: `public/index.html` (inline `:style` hexes, e.g. `:128` status-color fallback)
- Modify: `public/app.js` (any remaining inline hex, e.g. toast/badge helpers)

**Interfaces:**
- Consumes: tokens from Task 1.

- [ ] **Step 1: List remaining hexes**

```bash
grep -nE "#[0-9A-Fa-f]{3,6}" public/index.html public/app.js
```
This is the worklist. (Lines inside `STATUS_CONFIG/TYPE_CONFIG/RESULT_CONFIG` should
already be clean from Task 2.)

- [ ] **Step 2: Replace each per the mapping table**

For each line from Step 1, apply the mapping table. Specific known spot:
- `index.html:128` — `($store.app.STATUS_CONFIG[exp.status]?.dot || '#D…')` fallback →
  replace the literal with `'var(--status-archived)'`.
- SVG icon fills using `currentColor` need no change (they inherit `color`).
- Any `Flask` icon hardcoding `white` → leave as `white` (keyword, not hex; it sits
  on the accent logo box and is intentional).

- [ ] **Step 3: Full grep gate (the acceptance invariant)**

```bash
grep -nE "#[0-9A-Fa-f]{3,6}" public/index.html public/app.js
awk '/^:root/{r=1} /^}/{if(r){r=0; next}} !r' public/app.css | grep -nE "#[0-9A-Fa-f]{3,6}"
```
Expected: **no output from any of the three** (all hex now lives only in the
`app.css` `:root` token block). This satisfies the Global Constraints acceptance
invariant.

- [ ] **Step 4: Restart and verify**

Restart server (startup protocol) and reload. Expected: identical cohesive look,
no console errors.

---

## Task 5: Remove the "tweaks" panel and plumbing

Delete the per-user accent/background/density customization — the main source of
incoherence — and the parent-window edit-mode messaging it relied on.

**Files:**
- Modify: `public/index.html` (remove tweaks template `929-953`; fix body `:14`; fix density `:121`)
- Modify: `public/app.js` (remove store keys `301-302`, init plumbing `351-355`, `setTweaks` `437-442`)

**Interfaces:**
- Removes: `$store.app.tweaks`, `$store.app.tweaksPanelOpen`, `$store.app.setTweaks`,
  and the `__activate_edit_mode` / `__deactivate_edit_mode` / `__edit_mode_available`
  / `__edit_mode_set_keys` postMessage protocol. No other code may reference these
  after this task.

- [ ] **Step 1: Remove the tweaks panel template (index.html:929-953)**

Delete the entire block from `<template x-if="$store.app.tweaksPanelOpen">` through
its closing `</template>` (lines 929–953 inclusive).

- [ ] **Step 2: Fix the body element (index.html:14)**

Replace:
```html
<body x-data x-init="$store.app.init()" class="app-root" :style="'background:' + $store.app.tweaks.boardBg">
```
with (drop the `:style` binding; background now comes from the `--bg` token via CSS):
```html
<body x-data x-init="$store.app.init()" class="app-root">
```

- [ ] **Step 3: Fix the density binding (index.html:121)**

Replace the dynamic density class:
```html
<div class="column-cards" :class="'gap-' + $store.app.tweaks.density">
```
with a fixed compact density (a static class, no binding):
```html
<div class="column-cards gap-compact">
```

- [ ] **Step 4: Remove store keys (app.js:301-302)**

Delete these two lines:
```javascript
	tweaksPanelOpen: false,
	tweaks: { accentColor: '#2563EB', boardBg: '#F5F6F8', density: 'compact' },
```

- [ ] **Step 5: Remove edit-mode plumbing from `init()` (app.js:351-355)**

Delete the `window.addEventListener('message', …)` block and the
`window.parent.postMessage({ type: '__edit_mode_available' }, '*');` line
(the four lines spanning 351–355). Leave the rest of `init()` intact
(`loadSettings()`, `loadMetrics()`, etc.).

- [ ] **Step 6: Remove `setTweaks` (app.js:437-442)**

Delete the entire `setTweaks(updater) { … },` method.

- [ ] **Step 7: Grep gate — no tweaks references remain**

```bash
grep -nE "tweaks|boardBg|accentColor|edit_mode|__edit|setTweaks" public/index.html public/app.js
```
Expected: no output.

- [ ] **Step 8: Restart and verify**

Restart (startup protocol), reload. Expected: no tweaks panel, board background is
the token `--bg`, cards render at compact density, no console errors, no broken
Alpine bindings (check console for `tweaks is undefined`).

---

## Task 6: Component normalization & final acceptance

Tighten consistency now that everything is token-driven: unify buttons, status
pills, card/sidebar surfaces, and focus states. Then run the full acceptance gate.

**Files:**
- Modify: `public/app.css` (button/pill/surface/focus rules)

**Interfaces:**
- Consumes: all tokens; introduces no new ones (reuse only).

- [ ] **Step 1: Unify focus states**

Ensure every interactive element (`button`, `input`, `select`, `textarea`, `a.btn`,
`.tab`) shares one focus treatment. Add/normalize:
```css
button:focus-visible, input:focus-visible, select:focus-visible,
textarea:focus-visible, .tab:focus-visible {
	outline: none;
	box-shadow: 0 0 0 3px var(--focus-ring);
}
```

- [ ] **Step 2: Unify button variants**

Confirm there are exactly two button looks, both token-driven: primary
(`background: var(--accent); color: var(--surface)`, hover `var(--accent-hover)`)
and secondary/ghost (`background: var(--surface); color: var(--text-secondary);
border: 1px solid var(--border)`). Map any one-off button rules to one of these.

- [ ] **Step 3: Unify status-pill component**

Ensure all status/result pills use one base class with `color` + `background`
coming from the inline `var(--status-*)` / `var(--*-tint)` values (set in Task 2),
and identical radius/padding/font. Type badges use the neutral `.badge` look.

- [ ] **Step 4: Unify surfaces**

Cards, the detail sidebar, modals, and popovers all use `background: var(--surface)`,
`border: 1px solid var(--border)`; sunken areas use `var(--surface-sunken)`. The
app shell uses `var(--bg)`.

- [ ] **Step 5: Final acceptance gate (all checks)**

```bash
# (a) Acceptance invariant: no hex outside the :root token block
grep -nE "#[0-9A-Fa-f]{3,6}" public/index.html public/app.js
awk '/^:root/{r=1} /^}/{if(r){r=0; next}} !r' public/app.css | grep -nE "#[0-9A-Fa-f]{3,6}"
# (b) No tweaks remnants
grep -nE "tweaks|boardBg|accentColor|__edit|setTweaks" public/index.html public/app.js
```
Expected: all three commands produce no output.

- [ ] **Step 6: Full visual smoke test**

Restart (startup protocol). Walk every surface: Board, Timeline, List, experiment
detail sidebar (all tabs), creation wizard (all 4 steps), Settings, Metrics,
Populations, Owners pages. Confirm:
- One indigo accent; neutrals everywhere else.
- Color only on status/result elements + charts.
- Consistent buttons, pills, surfaces, focus rings.
- No console errors; no black/unstyled defaults.

Take a screenshot of the Board view for the record.

- [ ] **Step 7: Commit the plan/spec docs**

```bash
cd /home/rabt/devel/vstats
git add -f docs/superpowers/plans/2026-06-25-tracker-design-system.md
git commit -m "docs: tracker design-system implementation plan

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Token layer (primitives + semantic) → Task 1 + Token Reference. ✓
- 61-color migration → Tasks 2 (JS configs), 3 (app.css), 4 (inline), with grep gates. ✓
- Remove tweaks panel → Task 5. ✓
- Strict status-only color rule → enforced by neutralizing TYPE_CONFIG (Task 2) + mapping table "neutral wins" + documented rule comment (Task 1). ✓
- Component normalization → Task 6. ✓
- Light-first, dark-ready token structure → semantic/primitive split in Task 1. ✓
- Keep DM Sans/DM Mono → Global Constraints (fonts untouched). ✓
- Acceptance invariant (no hex outside `:root`) → grep gates in Tasks 3, 4, 6. ✓

**Placeholder scan:** No TBD/TODO; every edit shows exact before/after or exact code. Mapping table is exhaustive over the 60-color census. ✓

**Type/name consistency:** Token names used in Tasks 2–6 all match the Token Reference block (`--status-*`, `--status-*-tint`, `--accent`, `--accent-hover`, `--surface`, `--surface-sunken`, `--bg`, `--border`, `--border-strong`, `--text*`, `--success/warning/danger` + tints, `--focus-ring`). `STATUS_CONFIG/TYPE_CONFIG/RESULT_CONFIG` field names (`color/bg/dot/label`) unchanged from current code, so all existing `:style` consumers keep working. ✓
