# Cluster Sampling Calculators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two calculators — ICC Estimation and Design Effect (DEFF) — that let experimenters quantify within-cluster homogeneity and adjust sample sizes for cluster-randomized experiments.

**Architecture:** New `experiment/cluster.v` file adds three pure functions (`icc_estimate`, `design_effect`, `adjusted_sample_size`). Two new API endpoints (`/api/icc`, `/api/deff`) in `web/api_icc.v` and `web/api_deff.v` delegate to those functions. Two new HTML pages follow the existing template pattern. Frontend JS parses the CSV textarea client-side before sending structured JSON to the backend (consistent with how hypothesis tests work).

**Tech Stack:** V + veb for backend; Alpine.js for reactive UI; plain CSS using existing variables and classes.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `experiment/cluster.v` | Create | `icc_estimate`, `design_effect`, `adjusted_sample_size` |
| `tests/cluster_test.v` | Create | Unit tests for all three functions |
| `web/api_icc.v` | Create | `POST /api/icc` — decodes JSON, calls `icc_estimate` |
| `web/api_deff.v` | Create | `POST /api/deff` — decodes JSON, calls `design_effect` + `adjusted_sample_size` |
| `web/pages.v` | Modify | Add two page routes |
| `web/templates/_header.html` | Modify | Add two nav links |
| `web/templates/icc.html` | Create | ICC estimation page body |
| `web/templates/deff.html` | Create | DEFF adjustment page body |
| `web/static/js/calculators.js` | Modify | Add `iccCalc()`, `iccInterp()`, `deffCalc()`, `deffInterp()` |

---

## Task 1: Library functions (`experiment/cluster.v`)

**Files:**
- Create: `experiment/cluster.v`
- Create: `tests/cluster_test.v`

- [ ] **Step 1: Write failing tests**

Create `tests/cluster_test.v`:

```v
import experiment
import math

fn test__icc_well_separated_clusters() {
	// Groups with large between-cluster variance → high ICC
	groups := [
		[10.0, 11.0, 12.0, 13.0],
		[20.0, 21.0, 22.0, 23.0],
		[30.0, 31.0, 32.0, 33.0],
	]
	r := experiment.icc_estimate(groups, 0.05)
	assert r.rho > 0.98
	assert r.f_statistic > 100.0
	assert r.p_value < 0.05
	assert r.significant == true
	assert r.k == 3
	assert math.abs(r.avg_cluster_size - 4.0) < 0.001
}

fn test__icc_identical_group_means() {
	// Groups with identical means → near-zero or negative between-cluster variance
	groups := [
		[0.0, 2.0, 4.0],
		[0.0, 2.0, 4.0],
		[0.0, 2.0, 4.0],
	]
	r := experiment.icc_estimate(groups, 0.05)
	assert r.rho <= 0.0
	assert r.significant == false
}

fn test__design_effect_typical() {
	// 1 + (50-1)*0.12 = 1 + 5.88 = 6.88
	deff := experiment.design_effect(0.12, 50.0)
	assert math.abs(deff - 6.88) < 0.001
}

fn test__design_effect_zero_icc() {
	deff := experiment.design_effect(0.0, 100.0)
	assert math.abs(deff - 1.0) < 0.001
}

fn test__design_effect_max_icc() {
	// 1 + (10-1)*1 = 10
	deff := experiment.design_effect(1.0, 10.0)
	assert math.abs(deff - 10.0) < 0.001
}

fn test__adjusted_sample_size_exact() {
	// 1000 * 6.88 = 6880
	n := experiment.adjusted_sample_size(1000, 6.88)
	assert n == 6880
}

fn test__adjusted_sample_size_ceil() {
	// 100 * 1.503 = 150.3, ceil = 151
	n := experiment.adjusted_sample_size(100, 1.503)
	assert n == 151
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
v test tests/cluster_test.v
```

Expected: compile error — `experiment.icc_estimate` undefined.

- [ ] **Step 3: Implement `experiment/cluster.v`**

```v
module experiment

import math
import stats

pub struct ICCResult {
pub:
	rho              f64
	k                int
	avg_cluster_size f64
	f_statistic      f64
	p_value          f64
	significant      bool
}

// icc_estimate computes ICC(1) using one-way ANOVA decomposition with the
// m0 correction for unequal cluster sizes (Kish 1965).
pub fn icc_estimate(groups [][]f64, alpha f64) ICCResult {
	assert groups.len >= 2, 'ICC requires at least 2 clusters'
	k := groups.len
	mut n_total := 0
	mut n_sq_sum := 0.0
	for g in groups {
		assert g.len > 0, 'each cluster must have at least one observation'
		n_total += g.len
		n_sq_sum += f64(g.len * g.len)
	}
	mut sum_all := 0.0
	for g in groups {
		for v in g {
			sum_all += v
		}
	}
	grand_mean := sum_all / f64(n_total)
	mut ss_between := 0.0
	for g in groups {
		gm := stats.mean(g)
		ss_between += f64(g.len) * math.pow(gm - grand_mean, 2)
	}
	mut ss_within := 0.0
	for g in groups {
		gm := stats.mean(g)
		for v in g {
			ss_within += math.pow(v - gm, 2)
		}
	}
	df_between := k - 1
	df_within := n_total - k
	ms_between := if df_between > 0 { ss_between / f64(df_between) } else { 0.0 }
	ms_within := if df_within > 0 { ss_within / f64(df_within) } else { 0.0 }
	// Kish m0 correction handles unequal cluster sizes
	m0 := (1.0 / f64(k - 1)) * (f64(n_total) - n_sq_sum / f64(n_total))
	rho_denom := ms_between + (m0 - 1.0) * ms_within
	rho := if rho_denom > 0 { (ms_between - ms_within) / rho_denom } else { 0.0 }
	f_stat := if ms_within > 0 { ms_between / ms_within } else { 0.0 }
	p_value := if f_stat > 0 { 1.0 / (1.0 + f_stat) } else { 1.0 }
	mut avg_size := 0.0
	for g in groups {
		avg_size += f64(g.len)
	}
	avg_size /= f64(k)
	return ICCResult{
		rho:              rho
		k:                k
		avg_cluster_size: avg_size
		f_statistic:      f_stat
		p_value:          p_value
		significant:      p_value < alpha
	}
}

// design_effect returns the Kish design effect: 1 + (m - 1) * rho
pub fn design_effect(rho f64, m f64) f64 {
	return 1.0 + (m - 1.0) * rho
}

// adjusted_sample_size returns ceil(n * deff).
pub fn adjusted_sample_size(n int, deff f64) int {
	return int(math.ceil(f64(n) * deff))
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
v test tests/cluster_test.v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
make test
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add experiment/cluster.v tests/cluster_test.v
git commit -m "feat(experiment): add icc_estimate, design_effect, adjusted_sample_size"
```

---

## Task 2: API endpoints

**Files:**
- Create: `web/api_icc.v`
- Create: `web/api_deff.v`

- [ ] **Step 1: Create `web/api_icc.v`**

The JS frontend parses the CSV textarea and groups values by cluster_id before sending, so the API receives `groups [][]f64` (consistent with how hypothesis tests receive `[]f64` arrays).

```v
module main

import veb
import json
import vstats.experiment

struct ICCRequest {
pub mut:
	groups [][]f64
	alpha  f64 = 0.05
}

struct ICCResponse {
pub:
	rho              f64
	k                int
	avg_cluster_size f64
	f_statistic      f64
	p_value          f64
	significant      bool
}

@['/api/icc'; post]
pub fn (app &App) api_icc(mut ctx Context) veb.Result {
	req := json.decode(ICCRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.alpha <= 0 || req.alpha >= 1 {
		return api_error(mut ctx, 'alpha must be between 0 and 1')
	}
	if req.groups.len < 2 {
		return api_error(mut ctx, 'at least 2 clusters are required')
	}
	for g in req.groups {
		if g.len == 0 {
			return api_error(mut ctx, 'each cluster must have at least one observation')
		}
	}
	r := experiment.icc_estimate(req.groups, req.alpha)
	return ctx.json(ICCResponse{
		rho:              r.rho
		k:                r.k
		avg_cluster_size: r.avg_cluster_size
		f_statistic:      r.f_statistic
		p_value:          r.p_value
		significant:      r.significant
	})
}
```

- [ ] **Step 2: Create `web/api_deff.v`**

```v
module main

import veb
import json
import math
import vstats.experiment

struct DEFFRequest {
pub mut:
	rho f64
	m   f64
	n   int
}

struct DEFFResponse {
pub:
	deff            f64
	adjusted_n      int
	clusters_needed int
}

@['/api/deff'; post]
pub fn (app &App) api_deff(mut ctx Context) veb.Result {
	req := json.decode(DEFFRequest, ctx.req.data) or {
		return api_error(mut ctx, 'invalid JSON')
	}
	if req.rho < 0 || req.rho > 1 {
		return api_error(mut ctx, 'ICC (rho) must be between 0 and 1')
	}
	if req.m < 1 {
		return api_error(mut ctx, 'average cluster size must be at least 1')
	}
	if req.n < 1 {
		return api_error(mut ctx, 'sample size n must be at least 1')
	}
	deff := experiment.design_effect(req.rho, req.m)
	adjusted_n := experiment.adjusted_sample_size(req.n, deff)
	clusters_needed := int(math.ceil(f64(adjusted_n) / req.m))
	return ctx.json(DEFFResponse{
		deff:            deff
		adjusted_n:      adjusted_n
		clusters_needed: clusters_needed
	})
}
```

- [ ] **Step 3: Build to check for compile errors**

```bash
v build web/
```

Expected: exits 0, no errors.

- [ ] **Step 4: Commit**

```bash
git add web/api_icc.v web/api_deff.v
git commit -m "feat(web): add /api/icc and /api/deff endpoints"
```

---

## Task 3: Page routes and nav

**Files:**
- Modify: `web/pages.v`
- Modify: `web/templates/_header.html`

- [ ] **Step 1: Add routes to `web/pages.v`**

Append after the last route (`did_page`):

```v
@['/calculators/icc'; get]
pub fn (app &App) icc_page(mut ctx Context) veb.Result {
	html := read_template('icc') or { return ctx.server_error('template error') }
	return ctx.html(html)
}

@['/calculators/deff'; get]
pub fn (app &App) deff_page(mut ctx Context) veb.Result {
	html := read_template('deff') or { return ctx.server_error('template error') }
	return ctx.html(html)
}
```

- [ ] **Step 2: Add nav links to `web/templates/_header.html`**

In `_header.html`, the current last nav link is:
```html
  <a href="/calculators/did">DiD</a>
```

Add after it:
```html
  <a href="/calculators/icc">ICC</a>
  <a href="/calculators/deff">DEFF</a>
```

- [ ] **Step 3: Commit**

```bash
git add web/pages.v web/templates/_header.html
git commit -m "feat(web): add page routes and nav links for ICC and DEFF calculators"
```

---

## Task 4: ICC HTML template

**Files:**
- Create: `web/templates/icc.html`

- [ ] **Step 1: Create `web/templates/icc.html`**

```html
<h1>ICC Estimation</h1>
<p class="subtitle">Estimate the intraclass correlation from clustered experimental data</p>

<div x-data="iccCalc()" x-cloak>
  <div class="card">
    <div class="did-explainer">
      <p><strong>When to use:</strong> You have clustered data (users in accounts, students in classrooms, stores in regions) and need to quantify within-cluster homogeneity before designing a cluster-randomized experiment.</p>
      <p><strong>How it works:</strong> Decomposes outcome variance into between-cluster and within-cluster components via one-way ANOVA. A high ρ means cluster membership explains a large share of outcome variation.</p>
      <p class="did-example"><em>Example:</em> Paste account_id,revenue rows to see how homogeneous users within the same account are — which tells you how much your effective sample size shrinks when randomizing at the account level.</p>
    </div>

    <div class="field">
      <label>Data — <code>cluster_id, value</code> one row per line</label>
      <textarea x-model="f.data" rows="8" placeholder="store_1, 23.4&#10;store_1, 21.8&#10;store_2, 45.1&#10;store_2, 43.8"></textarea>
    </div>

    <div class="field-row">
      <div class="field">
        <label>Alpha</label>
        <input type="number" x-model.number="f.alpha" min="0.01" max="0.99" step="0.01">
      </div>
    </div>

    <div class="error-banner" x-show="error" x-text="error"></div>

    <div style="display:flex;gap:1rem;align-items:center">
      <button class="btn btn-primary" @click="submit()" :disabled="loading">
        <span x-show="loading" class="spinner"></span>
        <span x-text="loading ? 'Calculating...' : 'Calculate'"></span>
      </button>
      <button class="btn btn-secondary" @click="loadExample()">Load example</button>
    </div>
  </div>

  <div class="result-card" x-show="result">
    <div class="result-header">
      <h2>Result</h2>
      <span class="badge" :class="result && result.significant ? 'badge-sig' : 'badge-nosig'"
            x-text="result && result.significant ? 'Significant clustering' : 'No significant clustering'"></span>
    </div>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="label">ICC (ρ)</div>
        <div class="value" x-text="result ? fmt(result.rho) : '—'"></div>
      </div>
      <div class="stat-box">
        <div class="label">Clusters (k)</div>
        <div class="value" x-text="result ? result.k : '—'"></div>
      </div>
      <div class="stat-box">
        <div class="label">Avg cluster size</div>
        <div class="value" x-text="result ? fmt(result.avg_cluster_size, 1) : '—'"></div>
      </div>
      <div class="stat-box">
        <div class="label">F-statistic</div>
        <div class="value" x-text="result ? fmt(result.f_statistic) : '—'"></div>
      </div>
      <div class="stat-box">
        <div class="label">p-value</div>
        <div class="value" :class="result && result.significant ? 'sig' : 'nosig'" x-text="result ? fmt(result.p_value) : '—'"></div>
      </div>
    </div>
    <div class="interp" x-text="iccInterp(result)"></div>
  </div>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add web/templates/icc.html
git commit -m "feat(web): add ICC estimation page template"
```

---

## Task 5: DEFF HTML template

**Files:**
- Create: `web/templates/deff.html`

- [ ] **Step 1: Create `web/templates/deff.html`**

```html
<h1>Design Effect (DEFF)</h1>
<p class="subtitle">Adjust sample size for cluster-randomized experiments</p>

<div x-data="deffCalc()" x-cloak>
  <div class="card">
    <div class="did-explainer">
      <p><strong>When to use:</strong> You've computed a required n assuming simple random sampling (e.g., from the Power Analysis calculator), but your experiment randomizes by cluster (accounts, stores, schools). Multiply by DEFF to get the true required sample size.</p>
      <p><strong>Formula:</strong></p>
      <p class="did-formula">DEFF = 1 + (m̄ − 1) × ρ</p>
      <p class="did-example"><em>Example:</em> ρ = 0.12 and average cluster size 50 → DEFF = 6.88. An SRS n of 1000 becomes an adjusted n of 6880 — spread across 138 clusters.</p>
    </div>

    <div class="field-row">
      <div class="field">
        <label>ICC (ρ)</label>
        <input type="number" x-model.number="f.rho" min="0" max="1" step="0.01" placeholder="0.12">
      </div>
      <div class="field">
        <label>Avg cluster size (m̄)</label>
        <input type="number" x-model.number="f.m" min="1" step="1" placeholder="50">
      </div>
      <div class="field">
        <label>SRS sample size (n)</label>
        <input type="number" x-model.number="f.n" min="1" step="1" placeholder="1000">
      </div>
    </div>

    <div class="error-banner" x-show="error" x-text="error"></div>

    <div style="display:flex;gap:1rem;align-items:center">
      <button class="btn btn-primary" @click="submit()" :disabled="loading">
        <span x-show="loading" class="spinner"></span>
        <span x-text="loading ? 'Calculating...' : 'Calculate'"></span>
      </button>
      <button class="btn btn-secondary" @click="loadExample()">Load example</button>
    </div>
  </div>

  <div class="result-card" x-show="result">
    <div class="result-header">
      <h2>Result</h2>
    </div>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="label">DEFF</div>
        <div class="value" x-text="result ? fmt(result.deff, 3) : '—'"></div>
      </div>
      <div class="stat-box">
        <div class="label">Adjusted n</div>
        <div class="value" x-text="result ? result.adjusted_n : '—'"></div>
      </div>
      <div class="stat-box">
        <div class="label">Clusters needed</div>
        <div class="value" x-text="result ? result.clusters_needed : '—'"></div>
      </div>
    </div>
    <div class="interp" x-text="deffInterp(result)"></div>
  </div>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add web/templates/deff.html
git commit -m "feat(web): add DEFF adjustment page template"
```

---

## Task 6: Frontend JS components

**Files:**
- Modify: `web/static/js/calculators.js`

- [ ] **Step 1: Add ICC and DEFF components to `web/static/js/calculators.js`**

Append at the end of the file:

```javascript
function iccCalc() {
  return {
    f: { data: '', alpha: 0.05 },
    result: null,
    error: null,
    loading: false,

    loadExample() {
      this.result = null;
      this.error = null;
      var nl = String.fromCharCode(10);
      this.f.data = [
        'store_1,23.4','store_1,21.8','store_1,24.2','store_1,22.9','store_1,23.1',
        'store_2,45.1','store_2,43.8','store_2,46.2','store_2,44.5','store_2,45.7',
        'store_3,12.3','store_3,11.9','store_3,13.1','store_3,12.7','store_3,12.5',
        'store_4,67.2','store_4,65.8','store_4,68.4','store_4,66.9','store_4,67.5',
        'store_5,34.5','store_5,33.2','store_5,35.8','store_5,34.1','store_5,35.2',
        'store_6,89.3','store_6,87.6','store_6,90.1','store_6,88.8','store_6,89.7'
      ].join(nl);
      this.f.alpha = 0.05;
    },

    async submit() {
      this.error = null;
      this.result = null;
      this.loading = true;
      try {
        if (!this.f.data.trim()) throw new Error('Please enter data');
        var groups = parseICCData(this.f.data);
        this.result = await apiFetch('/api/icc', { groups: groups, alpha: this.f.alpha });
      } catch(e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    }
  };
}

function parseICCData(text) {
  var lines = text.trim().split('\n');
  var groupsMap = {};
  var order = [];
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i].trim();
    if (!line) continue;
    var parts = line.split(',');
    if (parts.length !== 2) throw new Error('Row ' + (i + 1) + ': expected cluster_id,value');
    var id = parts[0].trim();
    var val = parseFloat(parts[1].trim());
    if (!id) throw new Error('Row ' + (i + 1) + ': empty cluster_id');
    if (isNaN(val)) throw new Error('Row ' + (i + 1) + ': "' + parts[1].trim() + '" is not a number');
    if (!groupsMap[id]) { groupsMap[id] = []; order.push(id); }
    groupsMap[id].push(val);
  }
  if (order.length < 2) throw new Error('At least 2 clusters are required');
  return order.map(function(id) { return groupsMap[id]; });
}

function iccInterp(result) {
  if (!result) return '';
  var pct = (result.rho * 100).toFixed(1);
  var sig = result.significant
    ? 'The clustering effect is statistically significant (p=' + fmt(result.p_value) + ').'
    : 'The clustering effect is not statistically significant (p=' + fmt(result.p_value) + ').';
  return 'ρ = ' + fmt(result.rho) + ' — ' + pct + '% of outcome variance is explained by cluster membership. '
    + sig + ' Use this ρ in the Design Effect calculator to adjust your sample size for cluster-randomized experiments.';
}

function deffCalc() {
  return {
    f: { rho: null, m: null, n: null },
    result: null,
    error: null,
    loading: false,

    loadExample() {
      this.result = null;
      this.error = null;
      this.f.rho = 0.12;
      this.f.m = 50;
      this.f.n = 1000;
    },

    async submit() {
      this.error = null;
      this.result = null;
      this.loading = true;
      try {
        if (this.f.rho === null || this.f.rho < 0 || this.f.rho > 1)
          throw new Error('ICC (ρ) must be between 0 and 1');
        if (!this.f.m || this.f.m < 1)
          throw new Error('Average cluster size must be at least 1');
        if (!this.f.n || this.f.n < 1)
          throw new Error('Sample size n must be at least 1');
        this.result = await apiFetch('/api/deff', { rho: this.f.rho, m: this.f.m, n: this.f.n });
      } catch(e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    }
  };
}

function deffInterp(result) {
  if (!result) return '';
  return 'DEFF = ' + fmt(result.deff, 3) + ' — you need ' + fmt(result.deff, 2)
    + '× more subjects than simple random sampling requires. '
    + 'Enroll ' + result.adjusted_n + ' total subjects across '
    + result.clusters_needed + ' clusters.';
}
```

- [ ] **Step 2: Commit**

```bash
git add web/static/js/calculators.js
git commit -m "feat(web): add iccCalc and deffCalc Alpine components"
```

---

## Task 7: End-to-end verification

- [ ] **Step 1: Start the server**

```bash
v run web/main.v
```

Expected: server starts on port 8080.

- [ ] **Step 2: Verify ICC calculator**

Open `http://localhost:8080/calculators/icc`. Click **Load example**, then **Calculate**.

Expected results:
- ICC (ρ) ≈ 0.97–0.99 (stores have very different baseline revenue levels)
- k = 6
- Avg cluster size = 5.0
- p-value < 0.05, badge shows "Significant clustering"
- Interpretation text references ρ and suggests using the DEFF calculator

- [ ] **Step 3: Verify DEFF calculator**

Open `http://localhost:8080/calculators/deff`. Click **Load example** (ρ=0.12, m=50, n=1000), then **Calculate**.

Expected:
- DEFF = 6.880
- Adjusted n = 6880
- Clusters needed = 138

- [ ] **Step 4: Verify nav**

Both ICC and DEFF appear in the nav bar. Active page link is highlighted when on that page.

- [ ] **Step 5: Run full test suite one final time**

```bash
make test
```

Expected: all tests pass including the 7 new cluster tests.
