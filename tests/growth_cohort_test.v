module main

import growth

fn test_create_cohort_analysis() {
	// Test basic cohort analysis creation
	cohort_names := ['Jan', 'Feb', 'Mar']
	initial_sizes := [100, 100, 100]
	retention_data := [
		[100, 80, 60, 50],
		[100, 75, 55, 0],
		[100, 70, 0, 0],
	]

	ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)

	assert ca.cohorts.len == 3
	assert ca.retention_matrix[0][0] == 1.0      // 100/100
	assert ca.retention_matrix[0][1] == 0.80    // 80/100
	assert ca.retention_matrix[0][2] == 0.60    // 60/100
}

fn test_retention_at_period() {
	cohort_names := ['Jan', 'Feb']
	initial_sizes := [100, 100]
	retention_data := [
		[100, 80, 60],
		[100, 70, 50],
	]

	ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)

	assert ca.retention_at_period(0, 0) == 1.0
	assert ca.retention_at_period(0, 1) == 0.80
	assert ca.retention_at_period(0, 2) == 0.60
	assert ca.retention_at_period(1, 1) == 0.70
}

fn test_avg_retention_at_period() {
	cohort_names := ['Jan', 'Feb']
	initial_sizes := [100, 100]
	retention_data := [
		[100, 80],
		[100, 70],
	]

	ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)

	// Average of 0.80 and 0.70 = 0.75
	assert ca.avg_retention_at_period(1) == 0.75
}

fn test_churn_by_period() {
	cohort_names := ['Jan', 'Feb']
	initial_sizes := [100, 100]
	retention_data := [
		[100, 80, 60],
		[100, 70, 50],
	]

	ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)
	churn := ca.churn_by_period()

	// Churn from period 0 to 1: 1.0 - 0.75 = 0.25
	assert churn[0] == 0.25
}

fn test_compare_cohorts() {
	cohort_names := ['Jan', 'Feb', 'Mar']
	initial_sizes := [100, 100, 100]
	retention_data := [
		[100, 80, 60],
		[100, 70, 50],
		[100, 90, 80],
	]

	ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)

	ret_a, ret_b := ca.compare_cohorts('Jan', 'Mar')
	assert ret_a == 0.60  // Jan at period 2
	assert ret_b == 0.80  // Mar at period 2
}

fn test_cohort_with_revenue() {
	cohort_names := ['Jan', 'Feb']
	initial_sizes := [100, 100]
	retention_data := [
		[100, 80],
		[100, 75],
	]

	mut ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)

	revenue_data := [
		[0.0, 50.0],
		[0.0, 45.0],
	]
	ca.set_cohort_revenue(revenue_data)

	assert ca.cohorts.len == 2
}
