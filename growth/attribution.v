module growth

import math

pub struct AttributionResult {
pub mut:
	channel            string
	conversions        int
	revenue            f64
	attribution_score  f64
}

pub struct AttributionConfig {
pub mut:
	model     string
	half_life f64
}

pub fn first_touch_attributes(customer_channels []string, conversions []bool, revenue []f64) []AttributionResult {
	mut channel_results := map[string]AttributionResult{}
	
	for i in 0..customer_channels.len {
		channel := customer_channels[i]
		if channel !in channel_results {
			channel_results[channel] = AttributionResult{
				channel: channel
				conversions: 0
				revenue: 0.0
				attribution_score: 0.0
			}
		}
		
		if conversions[i] {
			channel_results[channel].conversions++
			channel_results[channel].revenue += revenue[i]
		}
	}
	
	return calculate_attribution_scores(channel_results.values().map(fn (r AttributionResult) AttributionResult { return r }))
}

pub fn last_touch_attributes(customer_channels []string, conversions []bool, revenue []f64) []AttributionResult {
	mut channel_results := map[string]AttributionResult{}
	mut last_channel := map[string]string{}
	
	for i in 0..customer_channels.len {
		customer_id := 'customer_${i}'
		last_channel[customer_id] = customer_channels[i]
	}
	
	for i in 0..customer_channels.len {
		if conversions[i] {
			customer_id := 'customer_${i}'
			channel := last_channel[customer_id]
			if channel !in channel_results {
				channel_results[channel] = AttributionResult{
					channel: channel
					conversions: 0
					revenue: 0.0
					attribution_score: 0.0
				}
			}
			channel_results[channel].conversions++
			channel_results[channel].revenue += revenue[i]
		}
	}
	
	return calculate_attribution_scores(channel_results.values().map(fn (r AttributionResult) AttributionResult { return r }))
}

pub fn linear_attributes(customer_touchpoints [][]string, conversions []bool, revenue []f64) []AttributionResult {
	mut channel_results := map[string]AttributionResult{}
	
	for i in 0..customer_touchpoints.len {
		if !conversions[i] { continue }
		
		touchpoints := customer_touchpoints[i]
		credit := revenue[i] / f64(touchpoints.len)
		
		for channel in touchpoints {
			if channel !in channel_results {
				channel_results[channel] = AttributionResult{
					channel: channel
					conversions: 0
					revenue: 0.0
					attribution_score: 0.0
				}
			}
			channel_results[channel].conversions++
			channel_results[channel].revenue += credit
		}
	}
	
	return calculate_attribution_scores(channel_results.values().map(fn (r AttributionResult) AttributionResult { return r }))
}

pub fn time_decay_attributes(customer_touchpoints [][]string, touchpoint_days [][]int, conversions []bool, revenue []f64, half_life f64) []AttributionResult {
	mut channel_results := map[string]AttributionResult{}
	
	for i in 0..customer_touchpoints.len {
		if !conversions[i] { continue }
		
		touchpoints := customer_touchpoints[i]
		days := touchpoint_days[i]
		mut total_weight := 0.0
		mut weights := []f64{}
		
		for j in 0..touchpoints.len {
			days_from_conversion := if j < days.len { days[j] } else { 0 }
			weight := math.pow(0.5, f64(days_from_conversion) / half_life)
			weights << weight
			total_weight += weight
		}
		
		for j in 0..touchpoints.len {
			channel := touchpoints[j]
			credit := revenue[i] * (weights[j] / total_weight)
			
			if channel !in channel_results {
				channel_results[channel] = AttributionResult{
					channel: channel
					conversions: 0
					revenue: 0.0
					attribution_score: 0.0
				}
			}
			channel_results[channel].conversions++
			channel_results[channel].revenue += credit
		}
	}
	
	return calculate_attribution_scores(channel_results.values().map(fn (r AttributionResult) AttributionResult { return r }))
}

pub fn position_based_attributes(customer_touchpoints [][]string, conversions []bool, revenue []f64) []AttributionResult {
	mut channel_results := map[string]AttributionResult{}
	
	for i in 0..customer_touchpoints.len {
		if !conversions[i] { continue }
		
		touchpoints := customer_touchpoints[i]
		n := touchpoints.len
		
		for j, channel in touchpoints {
			mut credit := revenue[i] * 0.20 / f64(n)
			
			if n == 1 {
				credit = revenue[i]
			} else if j == 0 {
				credit = revenue[i] * 0.40
			} else if j == n - 1 {
				credit = revenue[i] * 0.40
			}
			
			if channel !in channel_results {
				channel_results[channel] = AttributionResult{
					channel: channel
					conversions: 0
					revenue: 0.0
					attribution_score: 0.0
				}
			}
			channel_results[channel].conversions++
			channel_results[channel].revenue += credit
		}
	}
	
	return calculate_attribution_scores(channel_results.values().map(fn (r AttributionResult) AttributionResult { return r }))
}

fn calculate_attribution_scores(results []AttributionResult) []AttributionResult {
	mut total_revenue := 0.0
	for r in results {
		total_revenue += r.revenue
	}
	
	mut scored_results := results.clone()
	for i in 0..scored_results.len {
		scored_results[i].attribution_score = if total_revenue > 0 {
			scored_results[i].revenue / total_revenue
		} else { 0.0 }
	}
	
	return scored_results
}

pub fn channel_roi(attribution_results []AttributionResult, channel_costs map[string]f64) map[string]f64 {
	mut roi := map[string]f64{}
	for r in attribution_results {
		cost := if r.channel in channel_costs { channel_costs[r.channel] } else { 0.0 }
		roi[r.channel] = if cost > 0 { (r.revenue - cost) / cost } else { 0.0 }
	}
	return roi
}

pub fn optimal_channel_mix(channel_performance map[string]f64, total_budget f64) map[string]f64 {
	if channel_performance.len == 0 {
		return map[string]f64{}
	}
	
	mut total_perf := 0.0
	for _, perf in channel_performance {
		total_perf += perf
	}
	
	mut allocation := map[string]f64{}
	for channel, perf in channel_performance {
		allocation[channel] = if total_perf > 0 { (perf / total_perf) * total_budget } else { 0.0 }
	}
	
	return allocation
}

pub fn roas(revenue f64, ad_spend f64) f64 {
	if ad_spend == 0.0 {
		return 0.0
	}
	return revenue / ad_spend
}

pub fn blended_roas(total_revenue f64, total_ad_spend f64) f64 {
	return roas(total_revenue, total_ad_spend)
}
