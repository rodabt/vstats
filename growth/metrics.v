module growth

pub struct GrowthMetrics {
pub mut:
	arpa        f64
	arpu        f64
	cac         f64
	ltv         f64
	ltv_cac     f64
	churn_rate  f64
	retention   f64
	nrr         f64
}

pub fn calculate_growth_metrics(revenue f64, accounts int, users int, acquisition_spend f64, new_customers int, customers_lost int, mrr_start f64, mrr_end f64, churn_mrr f64, expansion_mrr f64, customer_lifespan f64) GrowthMetrics {
	return GrowthMetrics{
		arpa:        arpa(revenue, accounts)
		arpu:        arpu(revenue, users)
		cac:         cac(acquisition_spend, new_customers)
		ltv:         ltv(revenue, users, customer_lifespan)
		ltv_cac:     ltv_cac_ratio(revenue, users, customer_lifespan, acquisition_spend, new_customers)
		churn_rate:  churn_rate(customers_lost, accounts)
		retention:   retention_rate(customers_lost, accounts)
		nrr:         net_revenue_retention(mrr_start, mrr_end, churn_mrr, expansion_mrr)
	}
}

pub fn arpa(revenue f64, accounts int) f64 {
	if accounts == 0 {
		return 0.0
	}
	return revenue / f64(accounts)
}

pub fn arpu(revenue f64, users int) f64 {
	if users == 0 {
		return 0.0
	}
	return revenue / f64(users)
}

pub fn cac(acquisition_spend f64, new_customers int) f64 {
	if new_customers == 0 {
		return 0.0
	}
	return acquisition_spend / f64(new_customers)
}

pub fn ltv(revenue f64, users int, customer_lifespan f64) f64 {
	arpu_val := arpu(revenue, users)
	return arpu_val * customer_lifespan
}

pub fn ltv_cac_ratio(revenue f64, users int, customer_lifespan f64, acquisition_spend f64, new_customers int) f64 {
	cac_val := cac(acquisition_spend, new_customers)
	if cac_val == 0.0 {
		return 0.0
	}
	return ltv(revenue, users, customer_lifespan) / cac_val
}

pub fn churn_rate(customers_lost int, total_customers int) f64 {
	if total_customers == 0 {
		return 0.0
	}
	return f64(customers_lost) / f64(total_customers)
}

pub fn retention_rate(customers_lost int, total_customers int) f64 {
	return 1.0 - churn_rate(customers_lost, total_customers)
}

pub fn net_revenue_retention(mrr_start f64, mrr_end f64, churn_mrr f64, expansion_mrr f64) f64 {
	if mrr_start == 0.0 {
		return 0.0
	}
	return (mrr_end - churn_mrr) / mrr_start
}

pub fn gross_revenue_retention(mrr_start f64, churn_mrr f64) f64 {
	if mrr_start == 0.0 {
		return 0.0
	}
	return (mrr_start - churn_mrr) / mrr_start
}

pub fn payback_period(cac_ f64, monthly_arpu f64) f64 {
	if monthly_arpu == 0.0 {
		return 0.0
	}
	return cac_ / monthly_arpu
}

pub fn magic_number(net_new_arr f64, gross_margin f64, sales_marketing_spend f64) f64 {
	if sales_marketing_spend == 0.0 {
		return 0.0
	}
	return (net_new_arr * gross_margin) / sales_marketing_spend
}

pub fn rule_of_40(growth_rate f64, profit_margin f64) f64 {
	return growth_rate + profit_margin
}

pub fn monthly_recurring_revenue(plan_revenues []f64) f64 {
	mut total := 0.0
	for rev in plan_revenues {
		total += rev
	}
	return total
}

pub fn annual_recurring_revenue(mrr f64) f64 {
	return mrr * 12.0
}

pub fn customer_lifetime_value_simple(arpu f64, gross_margin f64, monthly_churn f64) f64 {
	if monthly_churn == 0.0 {
		return 0.0
	}
	return (arpu * gross_margin) / monthly_churn
}

pub fn burn_rate(starting_cash f64, ending_cash f64, months int) f64 {
	if months == 0 {
		return 0.0
	}
	return (starting_cash - ending_cash) / f64(months)
}

pub fn runway_months(current_cash f64, monthly_burn f64) f64 {
	if monthly_burn == 0.0 {
		return 0.0
	}
	return current_cash / monthly_burn
}
