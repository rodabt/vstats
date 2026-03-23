module growth

import math

pub struct CohortPeriod {
pub mut:
	period_index int
	cohort_size  int
	retained     int
	revenue      f64
	retention    f64
}

pub struct Cohort {
pub mut:
	name    string
	periods []CohortPeriod
}

pub struct CohortAnalysis {
pub mut:
	cohorts            []Cohort
	retention_matrix   [][]f64
	revenue_matrix     [][]f64
	avg_retention      []f64
	avg_revenue        []f64
	ltv_by_period      []f64
}

pub struct RetentionConfig {
pub mut:
	retention_type string
	cohort_type    string
}

pub fn create_cohort_analysis(cohort_names []string, initial_sizes []int, retention_data [][]int) CohortAnalysis {
	assert cohort_names.len == initial_sizes.len, "cohort names and sizes must match"
	assert cohort_names.len == retention_data.len, "cohort names and retention data must match"

	mut cohorts := []Cohort{}
	mut retention_matrix := [][]f64{}
	mut revenue_matrix := [][]f64{}

	for i in 0..cohort_names.len {
		mut periods := []CohortPeriod{}
		mut retention_row := []f64{}
		mut revenue_row := []f64{}

		for j in 0..retention_data[i].len {
			retention_rate := if initial_sizes[i] > 0 {
				f64(retention_data[i][j]) / f64(initial_sizes[i])
			} else { 0.0 }

			periods << CohortPeriod{
				period_index: j
				cohort_size:  initial_sizes[i]
				retained:     retention_data[i][j]
				revenue:      0.0
				retention:    retention_rate
			}
			retention_row << retention_rate
		}

		cohorts << Cohort{
			name:    cohort_names[i]
			periods: periods
		}
		retention_matrix << retention_row
		revenue_matrix << revenue_row
	}

	avg_retention := calculate_avg_by_period(retention_matrix)
	avg_revenue := calculate_avg_by_period(revenue_matrix)
	ltv_by_period := calculate_ltv_cumulative(revenue_matrix)

	return CohortAnalysis{
		cohorts:          cohorts
		retention_matrix: retention_matrix
		revenue_matrix:   revenue_matrix
		avg_retention:    avg_retention
		avg_revenue:      avg_revenue
		ltv_by_period:    ltv_by_period
	}
}

pub fn (mut ca CohortAnalysis) set_cohort_revenue(revenue_data [][]f64) {
	assert revenue_data.len == ca.cohorts.len, "revenue data must match cohort count"

	for i in 0..ca.cohorts.len {
		mut revenue_row := []f64{}
		for j in 0..revenue_data[i].len {
			if j < ca.cohorts[i].periods.len {
				ca.cohorts[i].periods[j].revenue = revenue_data[i][j]
			}
			revenue_row << revenue_data[i][j]
		}
		ca.revenue_matrix << revenue_row
	}
	ca.avg_revenue = calculate_avg_by_period(ca.revenue_matrix)
	ca.ltv_by_period = calculate_ltv_cumulative(ca.revenue_matrix)
}

fn calculate_avg_by_period(matrix [][]f64) []f64 {
	if matrix.len == 0 || matrix[0].len == 0 {
		return []f64{}
	}

	mut avg_by_period := []f64{len: matrix[0].len}
	for j in 0..matrix[0].len {
		mut sum := 0.0
		mut count := 0
		for i in 0..matrix.len {
			if j < matrix[i].len {
				sum += matrix[i][j]
				count++
			}
		}
		avg_by_period[j] = if count > 0 { sum / f64(count) } else { 0.0 }
	}
	return avg_by_period
}

fn calculate_ltv_cumulative(revenue_matrix [][]f64) []f64 {
	if revenue_matrix.len == 0 || revenue_matrix[0].len == 0 {
		return []f64{}
	}

	mut ltv_by_period := []f64{len: revenue_matrix[0].len}
	mut cumulative := 0.0
	for j in 0..revenue_matrix[0].len {
		for i in 0..revenue_matrix.len {
			if j < revenue_matrix[i].len {
				cumulative += revenue_matrix[i][j]
			}
		}
		ltv_by_period[j] = cumulative / f64(revenue_matrix.len)
	}
	return ltv_by_period
}

pub fn (ca CohortAnalysis) retention_at_period(cohort_index int, period int) f64 {
	if cohort_index >= ca.retention_matrix.len {
		return 0.0
	}
	if period >= ca.retention_matrix[cohort_index].len {
		return 0.0
	}
	return ca.retention_matrix[cohort_index][period]
}

pub fn (ca CohortAnalysis) avg_retention_at_period(period int) f64 {
	if period >= ca.avg_retention.len {
		return 0.0
	}
	return ca.avg_retention[period]
}

pub fn (ca CohortAnalysis) churn_by_period() []f64 {
	mut churn_rates := []f64{}
	for i in 1..ca.avg_retention.len {
		if ca.avg_retention[i - 1] > 0 {
			churn_rates << ca.avg_retention[i - 1] - ca.avg_retention[i]
		} else {
			churn_rates << 0.0
		}
	}
	return churn_rates
}

pub fn (ca CohortAnalysis) compare_cohorts(name_a string, name_b string) (f64, f64) {
	mut index_a := -1
	mut index_b := -1
	for i, c in ca.cohorts {
		if c.name == name_a { index_a = i }
		if c.name == name_b { index_b = i }
	}

	if index_a < 0 || index_b < 0 {
		return 0.0, 0.0
	}

	retention_a := if ca.retention_matrix[index_a].len > 0 {
		ca.retention_matrix[index_a][ca.retention_matrix[index_a].len - 1]
	} else { 0.0 }

	retention_b := if ca.retention_matrix[index_b].len > 0 {
		ca.retention_matrix[index_b][ca.retention_matrix[index_b].len - 1]
	} else { 0.0 }

	return retention_a, retention_b
}

pub fn (ca CohortAnalysis) ltv_projection(periods_to_project int, avg_revenue_per_user f64) []f64 {
	mut projected_ltv := []f64{}

	mut cumulative := 0.0
	max_observed := ca.avg_retention.len

	for i in 0..periods_to_project {
		if i < max_observed {
			cumulative += ca.avg_retention[i] * avg_revenue_per_user
		} else {
			mut last_churn := 0.0
			if max_observed > 1 {
				last_churn = ca.avg_retention[max_observed - 2] - ca.avg_retention[max_observed - 1]
			}
			projected_retention := if max_observed > 0 {
				ca.avg_retention[max_observed - 1] * math.pow(1.0 - last_churn, f64(i - max_observed + 1))
			} else { 0.0 }
			cumulative += projected_retention * avg_revenue_per_user
		}
		projected_ltv << cumulative
	}

	return projected_ltv
}

pub fn (ca CohortAnalysis) cohort_velocity(cohort_name string) int {
	for i, c in ca.cohorts {
		if c.name == cohort_name && i < ca.revenue_matrix.len {
			mut peak_period := 0
			mut peak_value := 0.0
			for j in 0..ca.revenue_matrix[i].len {
				if ca.revenue_matrix[i][j] > peak_value {
					peak_value = ca.revenue_matrix[i][j]
					peak_period = j
				}
			}
			return peak_period
		}
	}
	return -1
}
