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
    s: { control_mean: 12.5, control_std: 2.3, control_n: 1000, treatment_mean: 13.1, treatment_std: 2.4, treatment_n: 1000, alpha: 0.05 },
    raw: { control_data: '', treatment_data: '', alpha: 0.05 },
    result: null, error: null, loading: false,

    loadExample() {
      var nl = String.fromCharCode(10);
      this.result = null; this.error = null;
      if (this.metric === 'proportion') {
        this.p.successes_a = 480; this.p.n_a = 10000; this.p.successes_b = 530; this.p.n_b = 10000; this.p.alpha = 0.05;
      } else if (this.metric === 'summary') {
        this.s.control_mean = 12.5; this.s.control_std = 2.3; this.s.control_n = 1000;
        this.s.treatment_mean = 13.1; this.s.treatment_std = 2.4; this.s.treatment_n = 1000; this.s.alpha = 0.05;
      } else {
        this.raw.control_data  = ['12.1','11.8','13.4','12.0','11.5','12.8','11.9','12.3','11.7','12.6'].join(nl);
        this.raw.treatment_data = ['13.5','14.2','12.9','13.8','14.1','13.2','14.4','13.0','14.3','13.7'].join(nl);
        this.raw.alpha = 0.05;
      }
    },

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
        } else if (this.metric === 'summary') {
          var cm = this.s.control_mean, cs = this.s.control_std, cn = this.s.control_n;
          var tm = this.s.treatment_mean, ts = this.s.treatment_std, tn = this.s.treatment_n;
          var al = this.s.alpha;
          if (cn < 2 || tn < 2) throw new Error('Each group needs at least 2 observations');
          if (cs < 0 || ts < 0) throw new Error('Standard deviations must be non-negative');
          if (al <= 0 || al >= 1) throw new Error('Alpha must be between 0 and 1');
          payload = { metric: 'summary', control_mean: cm, control_std: cs, control_n: cn, treatment_mean: tm, treatment_std: ts, treatment_n: tn, alpha: al };
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

    loadExample() {
      this.result = null; this.error = null;
      this.f.successes_a = 480; this.f.n_a = 10000; this.f.successes_b = 530; this.f.n_b = 10000;
      this.f.mde = 0.01; this.f.alpha = 0.05; this.f.beta = 0.20;
    },

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

    loadExample() {
      this.result = null; this.error = null;
      if (this.metric === 'continuous') {
        this.c.effect_size = 0.2; this.c.alpha = 0.05; this.c.power = 0.8;
      } else {
        this.pr.p_baseline = 0.10; this.pr.p_treatment = 0.12; this.pr.alpha = 0.05; this.pr.power = 0.8;
      }
    },

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

    loadExample() {
      var nl = String.fromCharCode(10);
      this.result = null; this.error = null;
      this.f.control_post   = ['10.2','9.8','10.5','10.1','9.9'].join(nl);
      this.f.control_pre    = ['9.9','9.5','10.2','9.8','9.6'].join(nl);
      this.f.treatment_post = ['12.1','11.8','12.5','11.9','12.3'].join(nl);
      this.f.treatment_pre  = ['10.1','9.8','10.4','10.0','9.9'].join(nl);
      this.f.alpha = 0.05;
    },

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

    needsTwo() { return ['t_test_two_sample','mann_whitney','ks_test','correlation','wilcoxon','spearman_correlation'].indexOf(this.test) >= 0; },
    needsOne() { return ['t_test_one_sample','shapiro_wilk'].indexOf(this.test) >= 0; },

    testDesc() {
      var d = {
        t_test_two_sample: { when: 'Compare means of two independent groups. Best when data is approximately normal or n > 30.', ex: 'Are users from landing page A spending more time on-site than landing page B? Enter session durations for each group.' },
        t_test_one_sample: { when: 'Test whether a group\'s mean differs from a known benchmark (μ₀).', ex: 'Is average checkout time different from the target of 5 minutes? Enter times and set μ₀ = 5.' },
        mann_whitney:      { when: 'Non-parametric alternative to the two-sample t-test. Use when data is skewed, has outliers, or sample is small.', ex: 'Compare revenue per user between two variants when the distribution is right-skewed.' },
        ks_test:           { when: 'Tests whether two samples come from the same distribution — sensitive to differences in shape, not just means.', ex: 'Does the time-to-purchase distribution differ between mobile and desktop users?' },
        chi_squared:       { when: 'Tests independence between two categorical variables. Enter a contingency table — rows are groups, columns are outcomes.', ex: 'Did the new onboarding flow change the mix of free vs paid sign-ups? Row 1 = old, Row 2 = new; columns = free, paid.' },
        shapiro_wilk:      { when: 'Tests whether a sample is normally distributed. Run this before choosing between a t-test and Mann-Whitney.', ex: 'Paste your metric values. p < 0.05 means normality is rejected → use a non-parametric test.' },
        correlation:       { when: 'Tests whether two continuous variables are linearly associated (Pearson r). X and Y must be paired and the same length.', ex: 'Is session duration correlated with purchase value? Enter paired observations.' },
        wilcoxon:          { when: 'Paired non-parametric test for before/after measurements on the same units. X = before, Y = after.', ex: 'Did satisfaction scores improve after a redesign? Each user\'s before score in X, after score in Y.' },
        spearman_correlation: { when: 'Non-parametric test for monotonic association between two continuous variables. Use instead of Pearson when data is skewed, ordinal, or the relationship may be non-linear but monotonic. H₀: ρ_s = 0 (variables are independent).', ex: 'Do users who spend more time on the app tend to spend more money? Enter paired session-time and revenue values.' },
        runs_test:         { when: 'Tests whether the order of values in a sequence is random. Classifies each observation as above or below the median and counts "runs" (consecutive same-side sequences). H₀: the sequence is random. Use to detect trends, seasonality, or clustering over time.', ex: 'Daily conversion rates over 30 days: are they randomly scattered around the median, or do they show a drift?' }
      };
      return d[this.test] || { when: '', ex: '' };
    },

    loadExample() {
      var nl = String.fromCharCode(10);
      this.result = null; this.error = null;
      if (this.test === 't_test_two_sample') {
        this.f.x = ['4.2','5.1','3.8','4.7','5.3','4.0','4.9'].join(nl);
        this.f.y = ['5.8','6.2','5.4','6.7','5.9','6.1','6.5'].join(nl);
      } else if (this.test === 't_test_one_sample') {
        this.f.x = ['4.2','5.1','3.8','4.7','5.3','4.0','4.9'].join(nl); this.f.mu = 5.0;
      } else if (this.test === 'mann_whitney') {
        this.f.x = ['1.2','2.4','1.8','3.1','1.5','2.0'].join(nl);
        this.f.y = ['4.2','5.8','3.9','6.1','4.7','5.2'].join(nl);
      } else if (this.test === 'ks_test') {
        this.f.x = ['1.0','1.5','2.0','2.5','3.0','1.2'].join(nl);
        this.f.y = ['2.0','3.5','4.0','4.5','5.0','3.8'].join(nl);
      } else if (this.test === 'chi_squared') {
        this.f.contingency = '30, 10' + nl + '20, 40';
      } else if (this.test === 'shapiro_wilk') {
        this.f.x = ['4.2','5.1','3.8','4.7','5.3','4.0','4.9','5.5','4.3','4.8'].join(nl);
      } else if (this.test === 'correlation') {
        this.f.x = ['1','2','3','4','5','6','7'].join(nl);
        this.f.y = ['2.1','4.3','5.8','8.2','9.7','11.3','13.1'].join(nl);
      } else if (this.test === 'wilcoxon') {
        this.f.x = ['4.2','5.1','3.8','4.7','5.3','4.0'].join(nl);
        this.f.y = ['5.8','6.2','5.4','6.7','5.9','5.6'].join(nl);
      } else if (this.test === 'spearman_correlation') {
        this.f.x = ['2','4','4','4','5','7','9','10','10','12'].join(nl);
        this.f.y = ['1','2','3','5','5','6','8','9','10','12'].join(nl);
      } else if (this.test === 'runs_test') {
        this.f.x = ['1.2','3.5','2.1','4.8','3.2','5.1','4.4','6.2','5.8','7.1','6.3','8.0','7.2','9.1','8.5'].join(nl);
      }
    },

    async submit() {
      this.error = null; this.result = null; this.loading = true;
      try {
        var payload = { test: this.test, alpha: this.f.alpha };
        if (this.needsTwo()) {
          payload.x = parseCSV(this.f.x);
          payload.y = parseCSV(this.f.y);
          if (payload.x.length < 2 || payload.y.length < 2) throw new Error('Each group needs at least 2 observations');
          if ((this.test === 'wilcoxon' || this.test === 'spearman_correlation') && payload.x.length !== payload.y.length)
            throw new Error('This test requires equal-length paired arrays');
        } else if (this.needsOne()) {
          payload.x = parseCSV(this.f.x);
          if (payload.x.length < 3) throw new Error('Need at least 3 observations');
          payload.mu = this.f.mu;
        } else if (this.test === 'runs_test') {
          payload.x = parseCSV(this.f.x);
          if (payload.x.length < 10) throw new Error('Need at least 10 observations for the runs test');
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
  var labels = { t_test_two_sample:'t-statistic', t_test_one_sample:'t-statistic', mann_whitney:'U-statistic', ks_test:'D-statistic', chi_squared:'χ²-statistic', shapiro_wilk:'W-statistic', correlation:'r (Pearson)', wilcoxon:'W+', spearman_correlation:'Spearman ρ', runs_test:'Runs Z' };
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

    loadExample() {
      var nl = String.fromCharCode(10);
      this.result = null; this.error = null;
      this.f.treatment = ['0','0','0','0','0','1','1','1','1','1'].join(nl);
      this.f.y  = ['10.5','11.2','10.8','11.0','10.3','13.2','12.8','13.5','12.9','14.1'].join(nl);
      this.f.x  = ['28, 365','35, 730','42, 180','25, 90','31, 540',
                   '29, 400','36, 700','43, 200','26, 120','32, 500'].join(nl);
      this.f.caliper = -1; this.f.iterations = 1000;
    },

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
  var nl = String.fromCharCode(10);
  return {
    method: 'simple',
    alpha: 0.05,
    s: { y_treat_pre: '', y_treat_post: '', y_ctrl_pre: '', y_ctrl_post: '' },
    r: { y: '', group: '', time: '', x: '' },
    pt: { y_treated_pre: '', y_control_pre: '', time_pre: '' },
    ev: { y: '', group: '', relative_time: '' },
    result: null, error: null, loading: false,

    loadExample() {
      this.result = null; this.error = null;
      if (this.method === 'simple') {
        // 5 stores each; promotion raised treated by ~2 units, control unchanged
        this.s.y_treat_pre  = ['10.2','9.8','10.5','10.1','9.9'].join(nl);
        this.s.y_treat_post = ['12.1','11.8','12.5','11.9','12.3'].join(nl);
        this.s.y_ctrl_pre   = ['9.9','10.2','10.0','10.3','9.7'].join(nl);
        this.s.y_ctrl_post  = ['10.1','10.4','10.0','10.2','9.8'].join(nl);
      } else if (this.method === 'regression') {
        // 8 obs: ctrl pre/post × 2 units, treated pre/post × 2 units (2 repeats each)
        this.r.y     = ['10.0','10.2','9.9','10.1','9.8','12.1','9.9','12.3'].join(nl);
        this.r.group = ['0','0','0','0','1','1','1','1'].join(nl);
        this.r.time  = ['0','1','0','1','0','1','0','1'].join(nl);
        this.r.x = '';
      } else if (this.method === 'parallel') {
        // 4 pre-periods; both groups trend upward at similar rate
        this.pt.y_treated_pre = ['9.8','10.0','10.1','10.4'].join(nl);
        this.pt.y_control_pre = ['9.9','10.1','10.2','10.3'].join(nl);
        this.pt.time_pre      = ['1','2','3','4'].join(nl);
      } else if (this.method === 'event') {
        // 2 periods before & after; control obs interleaved with treated
        // 3 control units × 4 periods + 3 treated units × 4 periods = 24 rows
        var y   = [], grp = [], rt = [];
        var cPre  = [9.8, 10.0, 9.9],  cPost = [10.1, 10.2, 10.0];
        var tPre  = [10.1, 9.9, 10.2], tPost = [12.3, 12.0, 12.5];
        for (var t = -2; t <= 1; t++) {
          for (var u = 0; u < 3; u++) {
            var cVal = t < 0 ? cPre[u] + (t + 2) * 0.1 : cPost[u] + t * 0.1;
            y.push(cVal.toFixed(1)); grp.push('0'); rt.push(String(t));
          }
          for (var u = 0; u < 3; u++) {
            var tVal = t < 0 ? tPre[u] + (t + 2) * 0.1 : tPost[u] + t * 0.2;
            y.push(tVal.toFixed(1)); grp.push('1'); rt.push(String(t));
          }
        }
        this.ev.y             = y.join(nl);
        this.ev.group         = grp.join(nl);
        this.ev.relative_time = rt.join(nl);
      }
    },

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
