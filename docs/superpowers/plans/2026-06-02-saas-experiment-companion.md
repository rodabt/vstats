# SaaS Experiment Analytics Companion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a recipe-card codebook (`docs/companion/`) covering the full SaaS experiment lifecycle — ~35 cards across 8 Markdown files, each with When/Why/How/Pitfalls/Python(+chart)/SQL/vstats-placeholder.

**Architecture:** Flat recipe cards with a navigation index in README.md. Each file owns one thematic domain. Python examples are self-contained with synthetic data and Tufte-style matplotlib charts. SQL examples use four canonical SaaS tables.

**Tech Stack:** Markdown, Python 3, pandas, numpy, scipy.stats, statsmodels, matplotlib

---

### Task 0: README.md navigation index
- [ ] Create `docs/companion/README.md`

### Task 1: 00-foundations.md (4 cards)
- [ ] Counterfactual question
- [ ] Exchangeability / positivity / consistency
- [ ] Reading a DAG in SaaS
- [ ] Association vs. causation

### Task 2: 01-design.md (5 cards)
- [ ] Randomization unit
- [ ] Baseline and time zero
- [ ] Metric taxonomy
- [ ] Power analysis
- [ ] MDE to business value

### Task 3: 02-data-quality.md (6 cards)
- [ ] SRM
- [ ] ICC and contamination
- [ ] Network effects / interference
- [ ] Selection bias
- [ ] Novelty and primacy effects
- [ ] Survivor bias

### Task 4: 03-metric-pitfalls.md (5 cards)
- [ ] Simpson's paradox
- [ ] Skewness and heavy tails
- [ ] Ratio metrics
- [ ] Aggregation bias
- [ ] Regression to the mean

### Task 5: 04-variance-reduction.md (4 cards)
- [ ] CUPED
- [ ] Stratified randomization
- [ ] Winsorization
- [ ] Delta method

### Task 6: 05-causal-methods.md (5 cards)
- [ ] PSM
- [ ] DiD
- [ ] IV
- [ ] Doubly robust (AIPW)
- [ ] Target trial emulation

### Task 7: 06-readout.md (6 cards)
- [ ] HTE
- [ ] Multiple testing correction
- [ ] ITT vs per-protocol
- [ ] Mediation
- [ ] Sequential analysis (SPRT)
- [ ] Bayesian readout

### Task 8: 07-communication.md (4 cards)
- [ ] CIs in business language
- [ ] Caused vs. associated
- [ ] Experiment narrative
- [ ] Null results
