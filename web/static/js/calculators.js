// ── Shared helpers ────────────────────────────────────────────────────────────

function ciRange(lo, hi, dec) {
  dec = dec || 4;
  return '[' + fmt(lo, dec) + ', ' + fmt(hi, dec) + ']';
}

// ── A/B Test ──────────────────────────────────────────────────────────────────

function abTest() {
  return {
    metric: 'proportion',
    p: { successes_a: 500, n_a: 10000, successes_b: 550, n_b: 10000, alpha: 0.05 },
    raw: { control_data: '', treatment_data: '', alpha: 0.05 },
    result: null, error: null, loading: false,

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var payload;
        if (this.metric === 'proportion') {
          var sa = this.p.successes_a, na = this.p.n_a, sb = this.p.successes_b, nb = this.p.n_b, al = this.p.alpha;
          if (na < 2 || nb < 2) throw new Error('Group sizes must be at least 2');
          if (sa > na || sb > nb) throw new Error('Successes cannot exceed group size');
          if (al <= 0 || al >= 1) throw new Error('Alpha must be between 0 and 1');
          payload = { metric: 'proportion', successes_a: sa, n_a: na, successes_b: sb, n_b: nb, alpha: al };
        } else {
          var ctrl = parseCSV(this.raw.control_data);
          var trt  = parseCSV(this.raw.treatment_data);
          if (ctrl.length < 2 || trt.length < 2) throw new Error('Each group needs at least 2 observations');
          payload = { metric: 'continuous', control_data: ctrl, treatment_data: trt, alpha: this.raw.alpha };
        }
        this.result = await apiFetch('/api/ab-test', payload);
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    }
  };
}

function abInterp(result) {
  if (!result) return '';
  var p = fmt(result.p_value);
  var lift = pct(result.relative_lift);
  if (result.significant)
    return 'The difference is statistically significant (p=' + p + '). Treatment shows a ' + lift + ' relative lift over control.';
  return 'No statistically significant difference detected (p=' + p + '). The ' + lift + ' observed lift could be due to chance.';
}

// ── SPRT ──────────────────────────────────────────────────────────────────────

function sprtCalc() {
  return {
    f: { successes_a: 480, n_a: 10000, successes_b: 530, n_b: 10000, mde: 0.01, alpha: 0.05, beta: 0.20 },
    result: null, error: null, loading: false,

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var f = this.f;
        if (f.n_a < 1 || f.n_b < 1) throw new Error('Group sizes must be positive');
        if (f.mde <= 0) throw new Error('MDE must be positive');
        if (f.alpha <= 0 || f.alpha >= 1) throw new Error('Alpha must be between 0 and 1');
        if (f.beta <= 0 || f.beta >= 1) throw new Error('Beta must be between 0 and 1');
        this.result = await apiFetch('/api/sprt', { successes_a: f.successes_a, n_a: f.n_a, successes_b: f.successes_b, n_b: f.n_b, mde: f.mde, alpha: f.alpha, beta: f.beta });
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    }
  };
}

function sprtLabel(decision) {
  if (decision === 'reject_null')   return 'Reject H₀ (effect detected)';
  if (decision === 'accept_null')   return 'Accept H₀ (no effect)';
  return 'Continue Testing';
}

function sprtInterp(result) {
  if (!result) return '';
  var llr = fmt(result.log_likelihood_ratio, 3);
  if (result.decision === 'reject_null')
    return 'LLR=' + llr + ' crossed the upper boundary. Sufficient evidence to reject the null.';
  if (result.decision === 'accept_null')
    return 'LLR=' + llr + ' crossed the lower boundary. Insufficient evidence of an effect at the specified MDE.';
  return 'LLR=' + llr + ' is between the boundaries [' + fmt(result.lower_boundary,3) + ', ' + fmt(result.upper_boundary,3) + ']. Keep accumulating data.';
}

// ── Power Analysis ────────────────────────────────────────────────────────────

function powerCalc() {
  return {
    metric: 'continuous',
    c: { effect_size: 0.2, alpha: 0.05, power: 0.8 },
    pr: { p_baseline: 0.10, p_treatment: 0.12, alpha: 0.05, power: 0.8 },
    result: null, error: null, loading: false,

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var payload;
        if (this.metric === 'continuous') {
          if (this.c.effect_size <= 0) throw new Error('Effect size must be positive');
          payload = { metric: 'continuous', effect_size: this.c.effect_size, alpha: this.c.alpha, power: this.c.power };
        } else {
          var pb = this.pr.p_baseline, pt = this.pr.p_treatment;
          if (pb <= 0 || pb >= 1) throw new Error('Baseline rate must be between 0 and 1');
          if (pt <= 0 || pt >= 1) throw new Error('Target rate must be between 0 and 1');
          if (pb === pt) throw new Error('Baseline and target rates must differ');
          payload = { metric: 'proportion', p_baseline: pb, p_treatment: pt, alpha: this.pr.alpha, power: this.pr.power };
        }
        this.result = await apiFetch('/api/power-analysis', payload);
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    }
  };
}

function powerInterp(result) {
  if (!result) return '';
  var n = result.n_per_group;
  return 'You need ' + n.toLocaleString() + ' observations per group (' + (n * 2).toLocaleString() + ' total) to detect the specified effect.';
}

// ── CUPED ─────────────────────────────────────────────────────────────────────

function cupedCalc() {
  return {
    f: { control_post: '', control_pre: '', treatment_post: '', treatment_pre: '', alpha: 0.05 },
    result: null, error: null, loading: false,

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var cp = parseCSV(this.f.control_post);
        var cpr = parseCSV(this.f.control_pre);
        var tp = parseCSV(this.f.treatment_post);
        var tpr = parseCSV(this.f.treatment_pre);
        if (cp.length < 2 || tp.length < 2) throw new Error('Each group needs at least 2 observations');
        if (cp.length !== cpr.length) throw new Error('Control pre/post arrays must have the same length');
        if (tp.length !== tpr.length) throw new Error('Treatment pre/post arrays must have the same length');
        this.result = await apiFetch('/api/cuped', { control_post: cp, control_pre: cpr, treatment_post: tp, treatment_pre: tpr, alpha: this.f.alpha });
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    }
  };
}

function cupedInterp(result) {
  if (!result) return '';
  var vr = pct(result.variance_reduction);
  var p = fmt(result.adjusted ? result.adjusted.p_value : 0);
  var sig = result.adjusted && result.adjusted.significant;
  return 'CUPED reduced metric variance by ' + vr + '. Adjusted p-value: ' + p + '. ' + (sig ? 'The effect is statistically significant.' : 'No significant effect detected.');
}

// ── Hypothesis Tests ──────────────────────────────────────────────────────────

function hypothesisCalc() {
  return {
    test: 't_test_two_sample',
    f: { x: '', y: '', mu: 0, alpha: 0.05, contingency: '' },
    result: null, error: null, loading: false,

    needsTwo() { return ['t_test_two_sample','mann_whitney','ks_test','correlation','wilcoxon'].indexOf(this.test) >= 0; },
    needsOne() { return ['t_test_one_sample','shapiro_wilk'].indexOf(this.test) >= 0; },

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var payload = { test: this.test, alpha: this.f.alpha };
        if (this.needsTwo()) {
          payload.x = parseCSV(this.f.x);
          payload.y = parseCSV(this.f.y);
          if (payload.x.length < 2 || payload.y.length < 2) throw new Error('Each group needs at least 2 observations');
          if (this.test === 'wilcoxon' && payload.x.length !== payload.y.length)
            throw new Error('Wilcoxon requires equal-length paired arrays');
        } else if (this.needsOne()) {
          payload.x = parseCSV(this.f.x);
          if (payload.x.length < 3) throw new Error('Need at least 3 observations');
          payload.mu = this.f.mu;
        } else if (this.test === 'chi_squared') {
          var nl = String.fromCharCode(10);
          payload.contingency = this.f.contingency.trim().split(nl).map(function(row) {
            return row.split(',').map(function(v) {
              var n = parseInt(v.trim());
              if (isNaN(n)) throw new Error('Contingency values must be integers');
              return n;
            });
          });
        }
        this.result = await apiFetch('/api/hypothesis', payload);
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    }
  };
}

function hypStatLabel(test) {
  var labels = { t_test_two_sample:'t-statistic', t_test_one_sample:'t-statistic', mann_whitney:'U-statistic', ks_test:'D-statistic', chi_squared:'χ²-statistic', shapiro_wilk:'W-statistic', correlation:'r (Pearson)', wilcoxon:'W+' };
  return labels[test] || 'Statistic';
}

function hypInterp(result) {
  if (!result) return '';
  var p = fmt(result.p_value);
  return result.significant ? 'p=' + p + ' — reject H₀ at the given significance level.' : 'p=' + p + ' — insufficient evidence to reject H₀.';
}

// ── PSM ───────────────────────────────────────────────────────────────────────

function psmCalc() {
  return {
    f: { treatment: '', y: '', x: '', caliper: -1, iterations: 1000 },
    result: null, error: null, loading: false,

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var treatment = parseCSV(this.f.treatment);
        var y = parseCSV(this.f.y);
        if (treatment.length !== y.length) throw new Error('Treatment and Y must have the same number of units');
        var nl = String.fromCharCode(10);
        var xRows = this.f.x.trim().split(nl).map(function(row) { return parseCSV(row); });
        if (xRows.length !== treatment.length) throw new Error('X must have one row per unit');
        var ncols = xRows[0].length;
        if (!xRows.every(function(r) { return r.length === ncols; })) throw new Error('All X rows must have the same number of columns');
        if (!treatment.every(function(v) { return v === 0 || v === 1; })) throw new Error('Treatment must be binary (0 or 1)');
        this.result = await apiFetch('/api/psm', { treatment: treatment, y: y, x: xRows, caliper: this.f.caliper, iterations: this.f.iterations });
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    }
  };
}

function psmInterp(result) {
  if (!result || !result.ate || !result.balance) return '';
  var ate = fmt(result.ate.ate, 3);
  var p = fmt(result.ate.p_value);
  var bal = fmt(result.balance.mean_abs_smd_after, 3);
  var sig = result.ate.p_value < 0.05;
  return 'Estimated ATE=' + ate + ' (p=' + p + '). ' + (sig ? 'The treatment effect is statistically significant.' : 'No significant treatment effect.') + ' Mean absolute SMD after matching: ' + bal + ' (< 0.1 indicates good balance).';
}

// ── DiD ───────────────────────────────────────────────────────────────────────

function didCalc() {
  return {
    method: 'simple',
    alpha: 0.05,
    s: { y_treat_pre: '', y_treat_post: '', y_ctrl_pre: '', y_ctrl_post: '' },
    r: { y: '', group: '', time: '', x: '' },
    pt: { y_treated_pre: '', y_control_pre: '', time_pre: '' },
    ev: { y: '', group: '', relative_time: '' },
    result: null, error: null, loading: false,

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var nl = String.fromCharCode(10);
        var payload = { method: this.method, alpha: this.alpha };
        if (this.method === 'simple') {
          payload.y_treat_pre  = parseCSV(this.s.y_treat_pre);
          payload.y_treat_post = parseCSV(this.s.y_treat_post);
          payload.y_ctrl_pre   = parseCSV(this.s.y_ctrl_pre);
          payload.y_ctrl_post  = parseCSV(this.s.y_ctrl_post);
          if ([payload.y_treat_pre, payload.y_treat_post, payload.y_ctrl_pre, payload.y_ctrl_post].some(function(a) { return a.length < 2; }))
            throw new Error('Each group/period needs at least 2 observations');
        } else if (this.method === 'regression') {
          payload.y     = parseCSV(this.r.y);
          payload.group = parseCSV(this.r.group).map(function(v) { return Math.round(v); });
          payload.time  = parseCSV(this.r.time).map(function(v) { return Math.round(v); });
          if (payload.y.length !== payload.group.length || payload.y.length !== payload.time.length)
            throw new Error('Y, group, and time must have the same length');
          var xraw = this.r.x.trim();
          payload.x = xraw ? xraw.split(nl).map(function(row) { return parseCSV(row); }) : payload.y.map(function() { return []; });
        } else if (this.method === 'parallel') {
          payload.y_treated_pre = parseCSV(this.pt.y_treated_pre);
          payload.y_control_pre = parseCSV(this.pt.y_control_pre);
          payload.time_pre      = parseCSV(this.pt.time_pre).map(function(v) { return Math.round(v); });
        } else if (this.method === 'event') {
          payload.y             = parseCSV(this.ev.y);
          payload.group         = parseCSV(this.ev.group).map(function(v) { return Math.round(v); });
          payload.relative_time = parseCSV(this.ev.relative_time).map(function(v) { return Math.round(v); });
        }
        this.result = await apiFetch('/api/did', payload);
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    }
  };
}

function didIsSig(result, method) {
  if (!result) return false;
  var p = result.p_value !== undefined ? result.p_value : result.did_p_value;
  return p !== undefined && p < 0.05;
}

function didInterp(result, method) {
  if (!result) return '';
  if (method === 'simple') {
    return 'DiD effect=' + fmt(result.did_effect, 3) + ' (p=' + fmt(result.p_value) + '). ' + (didIsSig(result, method) ? 'The treatment caused a statistically significant change.' : 'No significant treatment effect detected.');
  }
  if (method === 'regression') {
    return 'DiD coefficient=' + fmt(result.did_coefficient, 3) + ' (p=' + fmt(result.did_p_value) + ', R²=' + fmt(result.r_squared, 3) + '). ' + (didIsSig(result, method) ? 'Significant effect.' : 'No significant effect.');
  }
  if (method === 'parallel') {
    return result.parallel_trends_hold
      ? 'Parallel trends assumption holds — no significant pre-trend difference detected. DiD is likely valid.'
      : 'Parallel trends may be violated (p=' + fmt(result.p_value) + '). Interpret DiD results with caution.';
  }
  return 'Event study complete. Review the per-period effects table above.';
}
