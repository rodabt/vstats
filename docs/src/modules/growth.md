# growth

`import vstats.growth`

Funnel analysis, cohort retention, marketing attribution, and SaaS metrics.

> **vs Python:** covers what would otherwise require custom pandas groupby for
> funnels, manual cohort loops, and ad-hoc attribution logic — no standard library
> covers this domain end-to-end.

## Funnels

```v
create_funnel(stage_names []string, stage_users []int) FunnelResult
// FunnelResult.conversion_rate f64
// FunnelResult.highest_drop_off() FunnelStage
// FunnelResult.lowest_drop_off() FunnelStage

ab_test_funnel(funnel_a FunnelResult, funnel_b FunnelResult) int  // 1=B wins, -1=A wins, 0=no diff
segment_funnel(segment_data map[string][]int) map[string]FunnelResult
projected_conversions(funnel FunnelResult, additional_users int) []int
```

## Cohort Retention

```v
create_cohort_analysis(cohort_names []string, initial_sizes []int, retention_data [][]int) CohortAnalysis

ca.retention_at_period(cohort_index int, period int) f64
ca.avg_retention_at_period(period int) f64
ca.churn_by_period() []f64
ca.compare_cohorts(name_a string, name_b string) (f64, f64)
ca.ltv_projection(periods_to_project int, avg_revenue_per_user f64) []f64
```

## Attribution

> **vs Python:** no standard library covers multi-touch attribution.
> `linear_attributes` splits credit equally across all touchpoints.
> `time_decay_attributes` weights recent touchpoints more heavily.

```v
first_touch_attributes(channels []string, conversions []bool, revenue []f64) []AttributionResult
last_touch_attributes(channels []string, conversions []bool, revenue []f64) []AttributionResult
linear_attributes(touchpoints [][]string, conversions []bool, revenue []f64) []AttributionResult
time_decay_attributes(touchpoints [][]string, touchpoint_days [][]int, conversions []bool, revenue []f64, half_life f64) []AttributionResult
position_based_attributes(touchpoints [][]string, conversions []bool, revenue []f64) []AttributionResult

channel_roi(attribution_results []AttributionResult, channel_costs map[string]f64) map[string]f64
optimal_channel_mix(channel_performance map[string]f64, total_budget f64) map[string]f64
roas(revenue f64, ad_spend f64) f64
```

## SaaS Metrics

```v
arpa(revenue f64, accounts int) f64
arpu(revenue f64, users int) f64
cac(acquisition_spend f64, new_customers int) f64
ltv(revenue f64, users int, customer_lifespan f64) f64
ltv_cac_ratio(revenue f64, users int, customer_lifespan f64, acquisition_spend f64, new_customers int) f64
churn_rate(customers_lost int, total_customers int) f64
retention_rate(customers_lost int, total_customers int) f64
net_revenue_retention(mrr_start f64, mrr_end f64, churn_mrr f64, expansion_mrr f64) f64
monthly_recurring_revenue(plan_revenues []f64) f64
annual_recurring_revenue(mrr f64) f64
rule_of_40(growth_rate f64, profit_margin f64) f64
burn_rate(starting_cash f64, ending_cash f64, months int) f64
runway_months(current_cash f64, monthly_burn f64) f64
```

## See Also

- [funnel-attribution example](../examples.html#funnel-attribution)
