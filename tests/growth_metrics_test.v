module main

import growth

fn test_arpa() {
	// Test basic ARPA calculation
	// Revenue: $100,000, 100 accounts -> ARPA = $1,000
	result := growth.arpa(100000.0, 100)
	assert result == 1000.0

	// ARPA with different values
	result2 := growth.arpa(50000.0, 50)
	assert result2 == 1000.0

	// ARPA with zero accounts
	result3 := growth.arpa(100000.0, 0)
	assert result3 == 0.0
}

fn test_arpu() {
	// Test basic ARPU calculation
	// Revenue: $100,000, 500 users -> ARPU = $200
	result := growth.arpu(100000.0, 500)
	assert result == 200.0

	// ARPU with zero users
	result2 := growth.arpu(100000.0, 0)
	assert result2 == 0.0
}

fn test_cac() {
	// Test basic CAC calculation
	// Spend: $10,000, 100 new customers -> CAC = $100
	result := growth.cac(10000.0, 100)
	assert result == 100.0

	// CAC with zero new customers
	result2 := growth.cac(10000.0, 0)
	assert result2 == 0.0
}

fn test_ltv() {
	// Test basic LTV calculation
	// ARPU: $100, Lifespan: 24 months -> LTV = $2,400
	result := growth.ltv(10000.0, 100, 24.0)
	assert result == 2400.0

	// LTV with zero users
	result2 := growth.ltv(10000.0, 0, 24.0)
	assert result2 == 0.0
}

fn test_ltv_cac_ratio() {
	// Test LTV:CAC ratio
	// LTV: $2,400, CAC: $100 -> Ratio = 24
	result := growth.ltv_cac_ratio(10000.0, 100, 24.0, 10000.0, 100)
	assert result == 24.0

	// Test with zero new customers (division by zero)
	result2 := growth.ltv_cac_ratio(10000.0, 100, 24.0, 10000.0, 0)
	assert result2 == 0.0
}

fn test_churn_rate() {
	// Test basic churn rate
	// 10 lost out of 100 -> 10% churn
	result := growth.churn_rate(10, 100)
	assert result == 0.10

	// Churn rate with zero total customers
	result2 := growth.churn_rate(10, 0)
	assert result2 == 0.0
}

fn test_retention_rate() {
	// Test retention rate
	// 10 lost out of 100 -> 90% retention
	result := growth.retention_rate(10, 100)
	assert result == 0.90

	// 100% retention when no customers lost
	result2 := growth.retention_rate(0, 100)
	assert result2 == 1.0
}

fn test_net_revenue_retention() {
	// Test NRR
	// MRR Start: $100,000, MRR End: $110,000, Churn MRR: $5,000, Expansion MRR: $15,000
	// NRR = ($110,000 - $5,000) / $100,000 = 105%
	result := growth.net_revenue_retention(100000.0, 110000.0, 5000.0, 15000.0)
	assert result == 1.05

	// NRR with zero MRR start
	result2 := growth.net_revenue_retention(0.0, 110000.0, 5000.0, 15000.0)
	assert result2 == 0.0
}

fn test_gross_revenue_retention() {
	// Test GRR
	// MRR Start: $100,000, Churn MRR: $5,000
	// GRR = ($100,000 - $5,000) / $100,000 = 95%
	result := growth.gross_revenue_retention(100000.0, 5000.0)
	assert result == 0.95

	// GRR with zero MRR start
	result2 := growth.gross_revenue_retention(0.0, 5000.0)
	assert result2 == 0.0
}

fn test_payback_period() {
	// Test payback period
	// CAC: $1,200, Monthly ARPU: $100 -> 12 months
	result := growth.payback_period(1200.0, 100.0)
	assert result == 12.0

	// Payback period with zero ARPU
	result2 := growth.payback_period(1200.0, 0.0)
	assert result2 == 0.0
}

fn test_magic_number() {
	// Test magic number
	// Net New ARR: $1,000,000, Gross Margin: 70%, S&M Spend: $500,000
	// Magic Number = ($1,000,000 × 0.70) / $500,000 = 1.4
	result := growth.magic_number(1000000.0, 0.70, 500000.0)
	assert result == 1.4

	// Magic number with zero S&M spend
	result2 := growth.magic_number(1000000.0, 0.70, 0.0)
	assert result2 == 0.0
}

fn test_rule_of_40() {
	// Test Rule of 40
	// Growth Rate: 30%, Profit Margin: 15% -> Score: 45
	result := growth.rule_of_40(30.0, 15.0)
	assert result == 45.0
}

fn test_monthly_recurring_revenue() {
	// Test MRR calculation
	// Three plans: $10,000 + $15,000 + $25,000 = $50,000
	plan_revenues := [10000.0, 15000.0, 25000.0]
	result := growth.monthly_recurring_revenue(plan_revenues)
	assert result == 50000.0

	// Empty plan list
	result2 := growth.monthly_recurring_revenue([]f64{})
	assert result2 == 0.0
}

fn test_annual_recurring_revenue() {
	// Test ARR calculation
	// MRR: $50,000 -> ARR: $600,000
	result := growth.annual_recurring_revenue(50000.0)
	assert result == 600000.0
}

fn test_customer_lifetime_value_simple() {
	// Test simple LTV
	// ARPU: $100, Gross Margin: 80%, Monthly Churn: 5%
	// LTV = ($100 × 0.80) / 0.05 = $1,600
	result := growth.customer_lifetime_value_simple(100.0, 0.80, 0.05)
	assert result == 1600.0

	// LTV with zero churn
	result2 := growth.customer_lifetime_value_simple(100.0, 0.80, 0.0)
	assert result2 == 0.0
}

fn test_burn_rate() {
	// Test burn rate
	// Starting: $1,000,000, Ending: $700,000, 6 months -> $50,000/month
	result := growth.burn_rate(1000000.0, 700000.0, 6)
	assert result == 50000.0

	// Burn rate with zero months
	result2 := growth.burn_rate(1000000.0, 700000.0, 0)
	assert result2 == 0.0
}

fn test_runway_months() {
	// Test runway calculation
	// Cash: $1,000,000, Monthly Burn: $50,000 -> 20 months
	result := growth.runway_months(1000000.0, 50000.0)
	assert result == 20.0

	// Runway with zero burn
	result2 := growth.runway_months(1000000.0, 0.0)
	assert result2 == 0.0
}

fn test_calculate_growth_metrics() {
	// Test calculating all metrics at once
	metrics := growth.calculate_growth_metrics(
		100000.0, 100, 500, 10000.0, 100, 10, 100000.0, 110000.0, 5000.0, 15000.0, 24.0
	)

	assert metrics.arpa == 1000.0
	assert metrics.arpu == 200.0
	assert metrics.cac == 100.0
	assert metrics.ltv == 4800.0
	assert metrics.churn_rate == 0.10
	assert metrics.retention == 0.90
}
