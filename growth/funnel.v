module growth

pub struct FunnelStage {
pub mut:
	name        string
	users       int
	conversions int
	dropouts    int
}

pub struct FunnelResult {
pub mut:
	stages          []FunnelStage
	conversion_rate f64
	drop_off_rate   f64
	total_conversion f64
}

pub struct FunnelConversion {
pub mut:
	from_name     string
	to_name       string
	from_users    int
	to_users      int
	rate          f64
	drop_off_rate f64
}

// create_funnel creates a funnel from stage data
// Each stage represents a step in the conversion funnel
pub fn create_funnel(stage_names []string, stage_users []int) FunnelResult {
	assert stage_names.len == stage_users.len, "stage_names and stage_users must have same length"
	assert stage_users.len >= 2, "funnel must have at least 2 stages"

	mut stages := []FunnelStage{}
	for i in 0..stage_names.len {
		mut conversions := 0
		if i < stage_users.len - 1 {
			conversions = if stage_users[i + 1] < stage_users[i] { stage_users[i + 1] } else { stage_users[i] }
		} else {
			conversions = stage_users[i]
		}

		stages << FunnelStage{
			name:        stage_names[i]
			users:       stage_users[i]
			conversions: conversions
			dropouts:    if i > 0 { stage_users[i - 1] - stage_users[i] } else { 0 }
		}
	}

	// Calculate overall conversion rate (first to last stage)
	total_conversion := if stage_users[0] > 0 {
		f64(stage_users[stage_users.len - 1]) / f64(stage_users[0])
	} else { 0.0 }

	// Calculate drop-off rate
	drop_off_rate := if stage_users[0] > 0 {
		1.0 - total_conversion
	} else { 0.0 }

	return FunnelResult{
		stages:          stages
		conversion_rate: total_conversion
		drop_off_rate:   drop_off_rate
		total_conversion: total_conversion
	}
}

// stage_conversion_rate calculates conversion rate between two consecutive stages
pub fn stage_conversion_rate(from_users int, to_users int) f64 {
	if from_users == 0 {
		return 0.0
	}
	return f64(to_users) / f64(from_users)
}

// stage_drop_off_rate calculates drop-off rate between two consecutive stages
pub fn stage_drop_off_rate(from_users int, to_users int) f64 {
	if from_users == 0 {
		return 0.0
	}
	return 1.0 - (f64(to_users) / f64(from_users))
}

// get_conversions returns detailed conversion data for each stage transition
pub fn (f FunnelResult) get_conversions() []FunnelConversion {
	mut conversions := []FunnelConversion{}
	for i in 0..f.stages.len - 1 {
		from := f.stages[i]
		to := f.stages[i + 1]
		conversions << FunnelConversion{
			from_name:     from.name
			to_name:       to.name
			from_users:    from.users
			to_users:      to.users
			rate:          stage_conversion_rate(from.users, to.users)
			drop_off_rate: stage_drop_off_rate(from.users, to.users)
		}
	}
	return conversions
}

// highest_drop_off returns the stage transition with the highest drop-off rate
pub fn (f FunnelResult) highest_drop_off() FunnelConversion {
	conversions := f.get_conversions()
	mut highest := conversions[0]
	for conv in conversions {
		if conv.drop_off_rate > highest.drop_off_rate {
			highest = conv
		}
	}
	return highest
}

// lowest_drop_off returns the stage transition with the lowest drop-off rate
pub fn (f FunnelResult) lowest_drop_off() FunnelConversion {
	conversions := f.get_conversions()
	mut lowest := conversions[0]
	for conv in conversions {
		if conv.drop_off_rate < lowest.drop_off_rate {
			lowest = conv
		}
	}
	return lowest
}

// funnel_value calculates the estimated value at each funnel stage
// Assumes linear value distribution between stages
pub fn funnel_value(stage_values []f64, stage_users []int) []f64 {
	assert stage_values.len == stage_users.len, "values and users must have same length"

	mut values := []f64{len: stage_values.len}
	for i in 0..stage_values.len {
		if stage_users[i] > 0 {
			values[i] = stage_values[i] / f64(stage_users[i])
		} else {
			values[i] = 0.0
		}
	}
	return values
}

// expected_revenue calculates expected revenue from funnel
// Formula: Sum of (conversion_rate × revenue_at_conversion) for each stage
pub fn expected_revenue(funnel FunnelResult, stage_revenues []f64) f64 {
	assert funnel.stages.len == stage_revenues.len, "funnel stages and revenues must match"

	mut total_revenue := 0.0
	for i in 0..funnel.stages.len {
		conversion_rate := if i == 0 { 1.0 } else {
			stage_conversion_rate(funnel.stages[i - 1].users, funnel.stages[i].users)
		}
		total_revenue += conversion_rate * stage_revenues[i]
	}
	return total_revenue
}

// ab_test_funnel compares two funnels to determine which performs better
// Returns the index of the better performing funnel (0 or 1)
pub fn ab_test_funnel(funnel_a FunnelResult, funnel_b FunnelResult) int {
	if funnel_a.total_conversion > funnel_b.total_conversion {
		return 0
	}
	return 1
}

// segment_funnel creates funnels for different user segments
// Useful for comparing conversion across demographics or behaviors
pub fn segment_funnel(segment_data map[string][]int) map[string]FunnelResult {
	mut segment_funnels := map[string]FunnelResult{}
	for segment, data in segment_data {
		mut stage_names := []string{}
		for i in 0..data.len {
			stage_names << "Stage ${i + 1}"
		}
		segment_funnels[segment] = create_funnel(stage_names, data)
	}
	return segment_funnels
}

// funnel_leakage calculates total users lost at each stage
pub fn funnel_leakage(funnel FunnelResult) []int {
	mut leakage := []int{}
	for i in 1..funnel.stages.len {
		leakage << funnel.stages[i - 1].users - funnel.stages[i].users
	}
	return leakage
}

// projected_conversions projects conversions if drop-off rates remain constant
pub fn projected_conversions(funnel FunnelResult, additional_users int) []int {
	mut projections := []int{}
	mut current_users := funnel.stages[funnel.stages.len - 1].users

	for i := funnel.stages.len - 2; i >= 0; i-- {
		drop_off := f64(funnel.stages[i].users - funnel.stages[i + 1].users)
		drop_off_rate := if funnel.stages[i].users > 0 {
			drop_off / f64(funnel.stages[i].users)
		} else { 0.0 }

		mut projected := current_users
		if drop_off_rate < 1.0 {
			projected = int(f64(current_users) / (1.0 - drop_off_rate))
		}
		if i == 0 {
			projected = current_users + additional_users
		}
		projections.prepend(projected)
		current_users = projected
	}
	return projections
}
