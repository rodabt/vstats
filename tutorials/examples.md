# VStats Examples & Applications

Real-world examples and case studies for data science and product analytics using VStats.

## Table of Contents

1. [Data Science Examples](#data-science-examples)
2. [Product Analytics Case Studies](#product-analytics-case-studies)
3. [Complete Workflows](#complete-workflows)

---

## Data Science Examples

### 1. Customer Churn Prediction

Predict which customers are likely to churn based on their behavior patterns.

```v
module main

import vstats.{stats, ml, utils}

fn main() {
    // Load customer data
    dataset := utils.load_breast_cancer() or {
        println('Error loading dataset: ${err}')
        return
    }

    // Split data: 80% train, 20% test
    split_idx := int(f64(dataset.features.len) * 0.8)
    x_train := dataset.features[0..split_idx]
    y_train := dataset.target[0..split_idx]
    x_test := dataset.features[split_idx..]
    y_test := dataset.target[split_idx..]

    // Normalize features for better model performance
    x_train_norm := normalize_features(x_train)
    x_test_norm := normalize_features(x_test)

    // Train logistic regression model
    mut model := ml.logistic_classifier(x_train_norm, y_train, 1000, 0.01)

    // Predict probabilities
    proba := ml.logistic_classifier_predict_proba(model, x_test_norm, 0.5)
    predictions := ml.logistic_classifier_predict(model, x_test_norm, 0.5)

    // Evaluate model performance
    cm := ml.confusion_matrix(y_test, predictions)
    println('Model Performance:')
    println('  Accuracy:  ${(cm.accuracy() * 100):.1f}%')
    println('  Precision: ${(cm.precision() * 100):.1f}%')
    println('  Recall:    ${(cm.recall() * 100):.1f}%')

    // Identify high-risk customers (probability > 0.7)
    println('\nHigh-Risk Customers (prob > 0.7):')
    for i, p in proba {
        if p > 0.7 {
            println('  Customer ${i}: ${(p * 100):.1f}% churn probability')
        }
    }
}

fn normalize_features(features [][]f64) [][]f64 {
    if features.len == 0 { return [][]f64{} }

    mut normalized := [][]f64{len: features.len}
    for j in 0..features[0].len {
        mut col := []f64{}
        for i in 0..features.len {
            col << features[i][j]
        }
        col_mean := stats.mean(col)
        col_std := stats.standard_deviation(col)
        if col_std > 0 {
            for i in 0..features.len {
                if j == 0 {
                    normalized[i] = []f64{}
                }
                normalized[i] << (features[i][j] - col_mean) / col_std
            }
        }
    }
    return normalized
}
```

### 2. Sales Forecasting with Linear Regression

Forecast future sales based on historical data.

```v
module main

import vstats.{ml, stats}

fn main() {
    // Historical monthly sales data (in thousands)
    months := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    sales := [45.0, 52.0, 48.0, 61.0, 55.0, 70.0, 68.0, 75.0, 82.0, 88.0, 92.0, 105.0]

    // Prepare features (add squared term for polynomial fit)
    mut x := [][]f64{}
    for m in months {
        x << [m, m * m]  // Linear and quadratic terms
    }

    // Train model
    model := ml.linear_regression(x, sales)

    // Forecast next 3 months
    future_months := [13.0, 14.0, 15.0]
    mut future_x := [][]f64{}
    for m in future_months {
        future_x << [m, m * m]
    }

    forecast := ml.linear_predict(model, future_x)

    println('Sales Forecast:')
    for i, m in future_months {
        println('  Month ${int(m)}: $${forecast[i]:.1f}K')
    }

    // Model evaluation
    predictions := ml.linear_predict(model, x)
    mse := ml.mse(sales, predictions)
    r2 := ml.r_squared(sales, predictions)

    println('\nModel Metrics:')
    println('  MSE: ${mse:.2f}')
    println('  R²:  ${r2:.4f}')
}
```

### 3. A/B Testing for Website Optimization

Test whether a new website design improves conversion rates.

```v
module main

import vstats.experiment

fn main() {
    // Simulated conversion data (time spent on page in seconds)
    control := [
        45.0, 52.0, 38.0, 61.0, 55.0, 48.0, 42.0, 59.0, 63.0, 51.0,
        47.0, 53.0, 40.0, 58.0, 49.0, 44.0, 56.0, 50.0, 62.0, 54.0,
        46.0, 41.0, 57.0, 43.0, 60.0, 48.0, 52.0, 45.0, 59.0, 55.0
    ]

    treatment := [
        68.0, 72.0, 65.0, 75.0, 71.0, 69.0, 74.0, 67.0, 78.0, 73.0,
        70.0, 66.0, 72.0, 68.0, 76.0, 71.0, 69.0, 74.0, 67.0, 72.0,
        75.0, 70.0, 73.0, 68.0, 71.0, 74.0, 69.0, 72.0, 70.0, 76.0
    ]

    // Run A/B test
    result := experiment.abtest(control, treatment, experiment.ABTestConfig{
        alpha: 0.05
    })

    println('A/B Test Results')
    println('================')
    println('Control Group:')
    println('  Mean: ${result.control_mean:.2f}s')
    println('  Std:  ${result.control_std:.2f}s')
    println('  N:    ${result.n_control}')

    println('\nTreatment Group:')
    println('  Mean: ${result.treatment_mean:.2f}s')
    println('  Std:  ${result.treatment_std:.2f}s')
    println('  N:    ${result.n_treatment}')

    println('\nStatistical Analysis:')
    println('  Relative Lift: ${(result.relative_lift * 100):.1f}%')
    println('  Effect Size:   ${result.effect_size:.3f}')
    println('  t-statistic:  ${result.t_statistic:.3f}')
    println('  p-value:      ${result.p_value:.4f}')
    println('  Significant:   ${result.significant}')

    println('\n95% Confidence Interval: [${result.ci_lower:.2f}, ${result.ci_upper:.2f}]')

    if result.significant {
        println('\nConclusion: The new design significantly improves user engagement.')
    }
}
```

### 4. User Segmentation with K-Means Clustering

Segment users into groups based on their behavior.

```v
module main

import vstats.{ml, stats}

fn main() {
    // User behavior data: [session_duration, pages_per_visit, purchases_per_month]
    user_data := [
        // High-value users
        [45.0, 12.0, 8.0],
        [52.0, 15.0, 10.0],
        [48.0, 11.0, 7.0],
        [55.0, 14.0, 9.0],
        // Medium-value users
        [25.0, 7.0, 3.0],
        [30.0, 8.0, 4.0],
        [22.0, 6.0, 2.0],
        [28.0, 9.0, 3.5],
        // Low-value users
        [8.0, 2.0, 0.5],
        [10.0, 3.0, 0.3],
        [12.0, 2.5, 0.4],
        [7.0, 2.0, 0.2],
        // Engaged but not purchasing
        [35.0, 18.0, 0.5],
        [40.0, 20.0, 0.8],
        [38.0, 16.0, 0.6],
        [42.0, 19.0, 0.7],
    ]

    // Normalize features
    normalized_data := normalize_data(user_data)

    // Cluster users into 4 segments
    model := ml.kmeans(normalized_data, 4, 100)

    println('User Segments')
    println('=============')

    mut segment_stats := map[int][]f64{}
    for i, label in model.labels {
        if label !in segment_stats {
            segment_stats[label] = []f64{}
        }
        segment_stats[label] << f64(i)
    }

    // Analyze each segment
    segment_names := ['High-Value', 'Medium-Value', 'Low-Value', 'Browsers']
    for seg_id in 0..4 {
        users := segment_stats[seg_id] or { continue }
        println('\nSegment ${seg_id} (${segment_names[seg_id]}):')
        println('  Users: ${users.len}')
        
        mut avg_duration := 0.0
        mut avg_pages := 0.0
        mut avg_purchases := 0.0
        for idx in users {
            avg_duration += user_data[idx][0]
            avg_pages += user_data[idx][1]
            avg_purchases += user_data[idx][2]
        }
        n := f64(users.len)
        println('  Avg Session: ${(avg_duration/n):.1f} min')
        println('  Avg Pages:   ${(avg_pages/n):.1f}')
        println('  Avg Purchases: ${(avg_purchases/n):.1f}/month')
    }

    // Calculate clustering quality
    silhouette := ml.silhouette_coefficient(normalized_data, model.labels)
    println('\nClustering Quality:')
    println('  Silhouette Score: ${silhouette:.3f}')
}

fn normalize_data(data [][]f64) [][]f64 {
    if data.len == 0 { return [][]f64{} }
    mut normalized := [][]f64{len: data.len}
    
    for j in 0..data[0].len {
        mut col := []f64{}
        for i in 0..data.len {
            col << data[i][j]
        }
        col_mean := stats.mean(col)
        col_std := stats.standard_deviation(col)
        if col_std > 0 {
            for i in 0..data.len {
                if j == 0 { normalized[i] = []f64{} }
                normalized[i] << (data[i][j] - col_mean) / col_std
            }
        }
    }
    return normalized
}
```

### 5. Statistical Analysis of Experiment Results

Analyze the results of a product experiment using hypothesis testing.

```v
module main

import vstats.{stats, hypothesis}

fn main() {
    // Pre-treatment baseline (monthly spending)
    baseline_a := [120.0, 115.0, 130.0, 125.0, 118.0, 122.0, 128.0, 117.0]
    baseline_b := [122.0, 118.0, 125.0, 130.0, 120.0, 124.0, 127.0, 119.0]

    // Post-treatment spending
    post_a := [125.0, 122.0, 135.0, 130.0, 123.0, 127.0, 133.0, 122.0]
    post_b := [145.0, 150.0, 155.0, 148.0, 152.0, 147.0, 158.0, 151.0]

    println('Statistical Analysis of Experiment Results')
    println('==========================================\n')

    // Check normality of distributions
    println('1. Normality Tests (Shapiro-Wilk)')
    stat_a, p_a := hypothesis.shapiro_wilk_test(post_a)
    stat_b, p_b := hypothesis.shapiro_wilk_test(post_b)
    println('   Group A: W=${stat_a:.4f}, p=${p_a:.4f} ${if p_a > 0.05 { "(Normal)" } else { "(Non-normal)" }}')
    println('   Group B: W=${stat_b:.4f}, p=${p_b:.4f} ${if p_b > 0.05 { "(Normal)" } else { "(Non-normal)" }}')

    // Descriptive statistics
    println('\n2. Descriptive Statistics')
    println('   Group A (Control):')
    println('     Mean: ${stats.mean(post_a):.2f}')
    println('     Std:  ${stats.standard_deviation(post_a):.2f}')
    println('     Median: ${stats.median(post_a):.2f}')
    
    println('   Group B (Treatment):')
    println('     Mean: ${stats.mean(post_b):.2f}')
    println('     Std:  ${stats.standard_deviation(post_b):.2f}')
    println('     Median: ${stats.median(post_b):.2f}')

    // Two-sample t-test
    println('\n3. Two-Sample t-Test')
    t_stat, p_val := hypothesis.t_test_two_sample(post_a, post_b, hypothesis.TestParams{})
    println('   t-statistic: ${t_stat:.4f}')
    println('   p-value:     ${p_val:.6f}')
    
    if p_val < 0.05 {
        println('   Result: Significant difference between groups (p < 0.05)')
    } else {
        println('   Result: No significant difference (p >= 0.05)')
    }

    // Effect size (Cohen's d)
    println('\n4. Effect Size (Cohen\'s d)')
    d := stats.cohens_d(post_a, post_b)
    println('   Cohen\'s d: ${d:.3f}')
    
    effect_interp := if math.abs(d) < 0.2 {
        'Negligible'
    } else if math.abs(d) < 0.5 {
        'Small'
    } else if math.abs(d) < 0.8 {
        'Medium'
    } else {
        'Large'
    }
    println('   Interpretation: ${effect_interp} effect')

    // Confidence interval for difference in means
    println('\n5. 95% Confidence Interval for Mean Difference')
    lower, upper := stats.confidence_interval_mean(post_a, 0.95)
    println('   Group A: [${lower:.2f}, ${upper:.2f}]')
    lower_b, upper_b := stats.confidence_interval_mean(post_b, 0.95)
    println('   Group B: [${lower_b:.2f}, ${upper_b:.2f}]')
}
```

---

## Product Analytics Case Studies

### 6. SaaS Metrics Dashboard

Track key SaaS metrics for a subscription business.

```v
module main

import vstats.growth

fn main() {
    println('SaaS Metrics Dashboard')
    println('======================\n')

    // Revenue metrics
    println('1. Revenue Metrics')
    mrr := growth.monthly_recurring_revenue([
        85000.0,   // Starter plan
        120000.0,  // Pro plan
        65000.0    // Enterprise plan
    ])
    println('   MRR: $${mrr:,.0f}')
    println('   ARR: $${growth.annual_recurring_revenue(mrr):,.0f}')

    accounts := 1250
    users := 3400
    arpa := growth.arpa(mrr, accounts)
    arpu := growth.arpu(mrr, users)
    println('   ARPA: $${arpa:.2f}')
    println('   ARPU: $${arpu:.2f}')

    // Customer acquisition
    println('\n2. Customer Acquisition')
    acquisition_spend := 85000.0
    new_customers := 125
    cac := growth.cac(acquisition_spend, new_customers)
    println('   New Customers: ${new_customers}')
    println('   Acquisition Spend: $${acquisition_spend:,.0f}')
    println('   CAC: $${cac:.2f}')

    // Lifetime value
    println('\n3. Customer Lifetime Value')
    avg_lifespan := 18.0  // months
    ltv := growth.ltv(mrr, users, avg_lifespan)
    ltv_cac := growth.ltv_cac_ratio(mrr, users, avg_lifespan, acquisition_spend, new_customers)
    println('   Avg Lifespan: ${avg_lifespan} months')
    println('   LTV: $${ltv:.2f}')
    println('   LTV:CAC Ratio: ${ltv_cac:.2f}x')

    payback := growth.payback_period(cac, arpu)
    println('   Payback Period: ${payback:.1f} months')

    // Retention
    println('\n4. Retention Metrics')
    total_customers := 1250
    customers_lost := 38
    churn := growth.churn_rate(customers_lost, total_customers)
    retention := growth.retention_rate(customers_lost, total_customers)
    println('   Churned Customers: ${customers_lost}')
    println('   Churn Rate: ${(churn * 100):.2f}%')
    println('   Retention Rate: ${(retention * 100):.2f}%')

    // Net Revenue Retention
    mrr_start := 100000.0
    mrr_end := 115000.0
    churn_mrr := 8000.0
    expansion_mrr := 23000.0
    nrr := growth.net_revenue_retention(mrr_start, mrr_end, churn_mrr, expansion_mrr)
    grr := growth.gross_revenue_retention(mrr_start, churn_mrr)
    println('\n5. Revenue Retention')
    println('   Net Revenue Retention: ${(nrr * 100):.1f}%')
    println('   Gross Revenue Retention: ${(grr * 100):.1f}%')

    // Growth efficiency
    println('\n6. Growth Efficiency')
    magic := growth.magic_number(180000.0, 0.75, 85000.0)
    println('   Magic Number: ${magic:.2f}')
    if magic > 1.0 {
        println('   Status: Efficient growth (<1 = burn too much)')
    }

    // Financial health
    println('\n7. Financial Health')
    burn := growth.burn_rate(2000000.0, 1500000.0, 6)
    runway := growth.runway_months(1500000.0, burn)
    println('   Monthly Burn: $${burn:,.0f}')
    println('   Runway: ${runway:.0f} months')
}
```

### 7. E-commerce Funnel Optimization

Analyze and optimize an e-commerce conversion funnel.

```v
module main

import vstats.growth

fn main() {
    println('E-commerce Funnel Analysis')
    println('==========================\n')

    // Full funnel: Visitor -> Add to Cart -> Checkout -> Purchase
    stage_names := ['Visitors', 'Product View', 'Add to Cart', 'Begin Checkout', 'Purchase']
    stage_users := [150000, 45000, 18000, 9000, 4500]

    funnel := growth.create_funnel(stage_names, stage_users)

    println('Funnel Overview:')
    println('  Total Visitors: ${stage_users[0]:,}')
    println('  Total Purchases: ${stage_users[4]:,}')
    println('  Overall Conversion: ${(funnel.total_conversion * 100):.2f}%')
    println('  Overall Drop-off: ${(funnel.drop_off_rate * 100):.1f}%\n')

    // Stage-by-stage analysis
    conversions := funnel.get_conversions()
    println('Stage-by-Stage Analysis:')
    println('-'.repeat(70))

    for conv in conversions {
        println('\n${conv.from_name} → ${conv.to_name}')
        println('  Conversion Rate: ${(conv.rate * 100):.1f}%')
        println('  Drop-off Rate:  ${(conv.drop_off_rate * 100):.1f}%')
        println('  Users Lost:     ${(conv.from_users - conv.to_users):,}')
    }

    // Identify problem areas
    worst := funnel.highest_drop_off()
    println('\n\nBiggest Opportunity:')
    println('  ${worst.from_name} → ${worst.to_name}')
    println('  ${(worst.drop_off_rate * 100):.1f}% drop-off')
    println('  ${(worst.from_users - worst.to_users):,} users lost')

    // Calculate revenue impact
    avg_order_value := 85.0
    potential_revenue := f64(stage_users[4] + (stage_users[0] - stage_users[1])) * avg_order_value
    current_revenue := f64(stage_users[4]) * avg_order_value
    
    println('\nRevenue Analysis:')
    println('  Current Revenue: $${current_revenue:,.0f}')
    println('  Potential (if 10% drop-off fixed): $${(potential_revenue * 1.1):,.0f}')
    println('  Revenue Gap: $${(potential_revenue * 1.1 - current_revenue):,.0f}')

    // Mobile vs Desktop comparison
    println('\n\nPlatform Comparison:')
    println('-'.repeat(70))

    platform_data := {
        'Desktop':   [100000, 35000, 14000, 7000, 3500]
        'Mobile':   [50000, 10000, 4000, 2000, 1000]
    }

    funnels := growth.segment_funnel(platform_data)

    for platform, f in funnels {
        println('\n${platform}:')
        println('  Conversion Rate: ${(f.total_conversion * 100):.2f}%')
        
        convs := f.get_conversions()
        for c in convs {
            if c.drop_off_rate > 0.6 {
                println('  ⚠ ${c.from_name} → ${c.to_name}: ${(c.drop_off_rate * 100):.0f}% drop-off')
            }
        }
    }
}
```

### 8. Marketing Attribution Analysis

Compare different attribution models to optimize marketing spend.

```v
module main

import vstats.growth

fn main() {
    println('Marketing Attribution Analysis')
    println('=============================\n')

    // Customer journeys with touchpoints
    customer_touchpoints := [
        ['Facebook', 'Google', 'Email', 'Direct'],
        ['Google', 'Google', 'Referral'],
        ['Facebook', 'Instagram', 'Google'],
        ['Organic', 'Email', 'Direct'],
        ['Google', 'Facebook', 'Email'],
        ['Referral', 'Direct'],
        ['Instagram', 'Facebook', 'Google', 'Email'],
        ['Google', 'Google', 'Direct'],
    ]

    conversions := [true, true, true, true, true, true, true, false]
    revenue := [150.0, 200.0, 180.0, 120.0, 175.0, 90.0, 220.0, 0.0]
    channel_costs := {
        'Facebook':   5000.0
        'Google':     8000.0
        'Instagram':   3000.0
        'Email':       1000.0
        'Organic':     500.0
        'Referral':    2000.0
        'Direct':      0.0
    }

    // First Touch Attribution
    println('1. First Touch Attribution')
    println('-'.repeat(40))
    first_touch := growth.first_touch_attributes(
        ['Facebook', 'Google', 'Facebook', 'Organic', 'Google', 'Referral', 'Instagram', 'Google'],
        conversions,
        revenue
    )

    mut first_total := 0.0
    for r in first_touch {
        first_total += r.revenue
        println('  ${r.channel}: $${r.revenue:.2f} (${(r.attribution_score * 100):.1f}%)')
    }
    println('  Total: $${first_total:.2f}\n')

    // Last Touch Attribution
    println('2. Last Touch Attribution')
    println('-'.repeat(40))
    last_touch := growth.last_touch_attributes(
        ['Direct', 'Referral', 'Email', 'Direct', 'Email', 'Direct', 'Email', 'Direct'],
        conversions,
        revenue
    )

    mut last_total := 0.0
    for r in last_touch {
        last_total += r.revenue
        println('  ${r.channel}: $${r.revenue:.2f} (${(r.attribution_score * 100):.1f}%)')
    }
    println('  Total: $${last_total:.2f}\n')

    // Linear Attribution
    println('3. Linear Attribution')
    println('-'.repeat(40))
    linear := growth.linear_attributes(customer_touchpoints, conversions, revenue)

    mut linear_total := 0.0
    for r in linear {
        linear_total += r.revenue
        println('  ${r.channel}: $${r.revenue:.2f} (${(r.attribution_score * 100):.1f}%)')
    }
    println('  Total: $${linear_total:.2f}\n')

    // Time Decay Attribution
    println('4. Time Decay Attribution (7-day half-life)')
    println('-'.repeat(40))
    touchpoint_days := [
        [14, 7, 3, 1],
        [30, 7, 1],
        [21, 14, 1],
        [7, 3, 1],
        [14, 3, 1],
        [1, 1],
        [21, 14, 7, 1],
        [7, 3, 1],
    ]
    time_decay := growth.time_decay_attributes(customer_touchpoints, touchpoint_days, conversions, revenue, 7.0)

    mut decay_total := 0.0
    for r in time_decay {
        decay_total += r.revenue
        println('  ${r.channel}: $${r.revenue:.2f} (${(r.attribution_score * 100):.1f}%)')
    }
    println('  Total: $${decay_total:.2f}\n')

    // ROI Analysis
    println('5. Channel ROI (Based on Linear Attribution)')
    println('-'.repeat(40))
    roi := growth.channel_roi(linear, channel_costs)

    for channel, value in roi {
        cost := channel_costs[channel] or { 0.0 }
        spend_pct := if first_total > 0 { (cost / first_total * 100) } else { 0.0 }
        println('  ${channel}:')
        println('    Spend: $${cost:,.0f}')
        println('    Revenue: $${(roi[channel] * cost + cost):,.0f}')
        println('    ROI: ${(value * 100):.1f}%')
    }

    // Blended ROAS
    total_revenue := first_total
    total_spend := 19500.0
    blended_roas := growth.blended_roas(total_revenue, total_spend)
    println('\n6. Blended ROAS')
    println('  Total Revenue: $${total_revenue:.2f}')
    println('  Total Spend: $${total_spend:,.0f}')
    println('  ROAS: ${blended_roas:.2f}x')
}
```

### 9. Cohort Retention Analysis

Track user retention over time using cohort analysis.

```v
module main

import vstats.growth

fn main() {
    println('Cohort Retention Analysis')
    println('========================\n')

    // Monthly cohorts with retention data
    cohort_names := ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    initial_sizes := [1000, 1100, 1200, 1150, 1300, 1400]

    // Retention by month (month 0, 1, 2, 3, 4, 5)
    retention_data := [
        [1000, 780, 620, 510, 430, 380],  // Jan cohort
        [1100, 850, 680, 560, 470, 0],    // Feb (no month 5)
        [1200, 920, 740, 610, 0, 0],      // Mar (no month 4,5)
        [1150, 880, 700, 0, 0, 0],        // Apr (no month 3,4,5)
        [1300, 1000, 0, 0, 0, 0],         // May (no month 2+)
        [1400, 0, 0, 0, 0, 0],            // Jun (only month 0)
    ]

    mut ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)

    // Add revenue data
    revenue_data := [
        [0.0, 39000, 37200, 33150, 30100, 28500],
        [0.0, 42500, 40800, 39200, 35250, 0],
        [0.0, 46000, 44400, 42700, 0, 0],
        [0.0, 44000, 42000, 0, 0, 0],
        [0.0, 50000, 0, 0, 0, 0],
        [0.0, 0, 0, 0, 0, 0],
    ]
    ca.set_cohort_revenue(revenue_data)

    // Print retention matrix
    println('Retention Matrix (%):')
    println('-'.repeat(70))
    print('Cohort    ')
    for i in 0..6 {
        print('  Month ${i}')
    }
    println('')
    println('-'.repeat(70))

    for i, cohort in ca.cohorts {
        print('${cohort.name:10}')
        for j in 0..6 {
            if j < ca.retention_matrix[i].len {
                print(' ${(ca.retention_matrix[i][j] * 100):5.1f}%')
            } else {
                print('      -')
            }
        }
        println('')
    }

    println('\nAverage Retention by Month:')
    println('-'.repeat(40))
    for i, avg in ca.avg_retention {
        bar_len := int(avg * 50)
        bar := '*'.repeat(bar_len)
        println('  Month ${i}: ${bar} ${(avg * 100):5.1f}%')
    }

    // Churn analysis
    println('\nMonthly Churn Rates:')
    println('-'.repeat(40))
    churn_rates := ca.churn_by_period()
    for i, churn in churn_rates {
        println('  Month ${i} → ${i+1}: ${(churn * 100):.1f}% churn')
    }

    // Cohort comparison
    println('\nCohort Comparison (Month 3 retention):')
    jan_ret, may_ret := ca.compare_cohorts('Jan', 'May')
    println('  Jan cohort: ${(jan_ret * 100):.1f}%')
    println('  May cohort: ${(may_ret * 100):.1f}%')
    
    improvement := (may_ret - jan_ret) / jan_ret * 100
    if improvement > 0 {
        println('  📈 Retention improved by ${improvement:.1f}%')
    } else {
        println('  📉 Retention decreased by ${-improvement:.1f}%')
    }

    // LTV projection
    println('\nCumulative LTV by Month:')
    println('-'.repeat(40))
    for i, ltv in ca.ltv_by_period {
        if ltv > 0 {
            println('  Month ${i}: $${ltv:.2f}')
        }
    }
}
```

### 10. Time Series Anomaly Detection

Detect anomalies in time series data using statistical methods.

```v
module main

import vstats.{stats, hypothesis}

fn main() {
    // Daily active users for the past 30 days
    dau := [
        1250.0, 1280.0, 1310.0, 1295.0, 1270.0,
        1300.0, 1320.0, 1290.0, 1315.0, 1285.0,
        1275.0, 1305.0, 1325.0, 1310.0, 1295.0,
        1280.0, 1300.0, 1315.0, 1650.0, 1320.0,  // Day 19 is anomalous!
        1305.0, 1290.0, 1310.0, 1280.0, 1275.0,
        1295.0, 1300.0, 1310.0, 1290.0, 1285.0
    ]

    println('Time Series Anomaly Detection')
    println('============================\n')

    // Calculate baseline statistics (excluding potential anomaly)
    baseline := dau[0..18]
    baseline_mean := stats.mean(baseline)
    baseline_std := stats.standard_deviation(baseline)

    println('Baseline Statistics (Days 1-18):')
    println('  Mean: ${baseline_mean:.1f}')
    println('  Std Dev: ${baseline_std:.1f}')

    // Set anomaly threshold (2 standard deviations)
    threshold_upper := baseline_mean + 2 * baseline_std
    threshold_lower := baseline_mean - 2 * baseline_std

    println('\nAnomaly Thresholds:')
    println('  Upper: ${threshold_upper:.1f}')
    println('  Lower: ${threshold_lower:.1f}')

    // Detect anomalies
    println('\nAnomaly Detection Results:')
    println('-'.repeat(40))

    mut anomalies := []int{}
    for i, val in dau {
        if val > threshold_upper || val < threshold_lower {
            anomalies << i + 1
            deviation := (val - baseline_mean) / baseline_std
            println('  Day ${i + 1}: ${val:.0f} (${deviation:+.2f} std devs)')
            println('    ${if val > threshold_upper { "Above threshold" } else { "Below threshold" }}')
        }
    }

    if anomalies.len == 0 {
        println('  No anomalies detected')
    }

    // Moving average analysis
    println('\n7-Day Moving Average:')
    println('-'.repeat(40))

    window_size := 7
    for i in window_size - 1..dau.len {
        window := dau[i - window_size + 1..i + 1]
        ma := stats.mean(window)
        deviation := (dau[i] - ma) / stats.standard_deviation(window)
        
        if math.abs(deviation) > 1.5 {
            println('  Day ${i + 1}: MA=${ma:.1f}, Actual=${dau[i]:.0f}, z=${deviation:+.2f} ⚠')
        }
    }

    // Trend analysis
    println('\nTrend Analysis:')
    println('-'.repeat(40))

    first_half := stats.mean(dau[0..15])
    second_half := stats.mean(dau[15..])
    trend_change := (second_half - first_half) / first_half * 100

    println('  First half avg: ${first_half:.1f}')
    println('  Second half avg: ${second_half:.1f}')
    println('  Change: ${trend_change:+.1f}%')

    if math.abs(trend_change) > 5 {
        println('  Significant trend ${if trend_change > 0 { "upward" } else { "downward" }} detected')
    }
}
```

---

## Complete Workflows

### 11. End-to-End Machine Learning Pipeline

```v
module main

import vstats.{linalg, stats, ml, utils, optim, nn}

fn main() {
    println('ML Pipeline: Titanic Survival Prediction')
    println('======================================\n')

    // 1. Data Loading
    println('Step 1: Loading Data')
    dataset := utils.load_titanic() or {
        println('Error: ${err}')
        return
    }
    println('  Loaded ${dataset.features.len} samples with ${dataset.features[0].len} features')

    // 2. Exploratory Data Analysis
    println('\nStep 2: Exploratory Data Analysis')
    analyze_dataset(dataset)

    // 3. Train/Test Split
    println('\nStep 3: Train/Test Split')
    split_idx := int(f64(dataset.features.len) * 0.8)
    x_train := dataset.features[0..split_idx]
    y_train := dataset.target[0..split_idx]
    x_test := dataset.features[split_idx..]
    y_test := dataset.target[split_idx..]
    println('  Training samples: ${x_train.len}')
    println('  Test samples: ${x_test.len}')

    // 4. Feature Normalization
    println('\nStep 4: Feature Normalization')
    x_train_norm := normalize_matrix(x_train)
    x_test_norm := normalize_matrix(x_test)
    println('  Features normalized to zero mean, unit variance')

    // 5. Model Training
    println('\nStep 5: Model Training')
    
    // Logistic Regression
    println('  Training Logistic Regression...')
    lr_model := ml.logistic_classifier(x_train_norm, y_train, 1000, 0.01)
    lr_pred := ml.logistic_classifier_predict(lr_model, x_test_norm, 0.5)
    lr_acc := ml.accuracy(y_test, lr_pred)
    println('  Logistic Regression Accuracy: ${(lr_acc * 100):.1f}%')

    // Random Forest
    println('  Training Random Forest...')
    rf_model := ml.random_forest_classifier(x_train, y_train, 50, 8)
    rf_pred := ml.random_forest_classifier_predict(rf_model, x_test)
    rf_acc := ml.accuracy(y_test, rf_pred)
    println('  Random Forest Accuracy: ${(rf_acc * 100):.1f}%')

    // 6. Model Selection
    println('\nStep 6: Model Selection')
    best_model := if rf_acc > lr_acc { 'Random Forest' } else { 'Logistic Regression' }
    best_pred := if rf_acc > lr_acc { rf_pred } else { lr_pred }
    best_acc := if rf_acc > lr_acc { rf_acc } else { lr_acc }
    println('  Best Model: ${best_model}')
    println('  Best Accuracy: ${(best_acc * 100):.1f}%')

    // 7. Detailed Evaluation
    println('\nStep 7: Detailed Evaluation')
    cm := ml.confusion_matrix(y_test, best_pred)
    println('  Confusion Matrix:')
    println('                   Predicted')
    println('                 Not Survive  Survive')
    println('  Actual Not Survive    ${cm[0][0]:3}        ${cm[0][1]:3}')
    println('  Actual Survive        ${cm[1][0]:3}        ${cm[1][1]:3}')
    println('  Accuracy:  ${(cm.accuracy() * 100):.1f}%')
    println('  Precision: ${(cm.precision() * 100):.1f}%')
    println('  Recall:    ${(cm.recall() * 100):.1f}%')
    println('  F1 Score:  ${(cm.f1_score() * 100):.1f}%')

    // 8. Feature Importance (for Random Forest)
    if best_model == 'Random Forest' {
        println('\nStep 8: Feature Importance')
        feature_names := ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        println('  Feature Analysis:')
        for i in 0..feature_names.len {
            mut feature_vals := []f64{}
            for sample in x_test {
                feature_vals << sample[i]
            }
            corr := stats.correlation(feature_vals, best_pred.map(f64(it)))
            println('    ${feature_names[i]:10} correlation: ${corr:+.3f}')
        }
    }
}

fn analyze_dataset(dataset utils.Dataset) {
    mut feature_cols := [][]f64{}
    for i in 0..dataset.features[0].len {
        mut col := []f64{}
        for sample in dataset.features {
            col << sample[i]
        }
        feature_cols << col
    }

    println('  Feature Statistics:')
    feature_names := ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    for i, col in feature_cols {
        if i < feature_names.len {
            println('    ${feature_names[i]:10} - Mean: ${stats.mean(col):6.1f}, Std: ${stats.standard_deviation(col):6.1f}')
        }
    }

    // Target distribution
    mut survived := 0
    for label in dataset.target {
        if label == 1 { survived++ }
    }
    survival_rate := f64(survived) / f64(dataset.target.len)
    println('\n  Target Distribution:')
    println('    Survived: ${survived} (${(survival_rate * 100):.1f}%)')
    println('    Not Survived: ${dataset.target.len - survived} (${((1 - survival_rate) * 100):.1f}%)')
}

fn normalize_matrix(data [][]f64) [][]f64 {
    if data.len == 0 { return [][]f64{} }
    mut normalized := [][]f64{len: data.len}
    
    for j in 0..data[0].len {
        mut col := []f64{}
        for i in 0..data.len {
            col << data[i][j]
        }
        col_mean := stats.mean(col)
        col_std := stats.standard_deviation(col)
        if col_std > 0 {
            for i in 0..data.len {
                if j == 0 { normalized[i] = []f64{} }
                normalized[i] << (data[i][j] - col_mean) / col_std
            }
        }
    }
    return normalized
}
```

---

## Appendix: Common Formulas

### Statistics
- **Mean**: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$
- **Variance**: $\sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$
- **Standard Deviation**: $\sigma = \sqrt{\sigma^2}$
- **Correlation**: $r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}$

### Machine Learning
- **MSE**: $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE**: $\sqrt{MSE}$
- **MAE**: $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **R²**: $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

### Growth Metrics
- **ARPA**: $\frac{\text{Revenue}}{\text{Accounts}}$
- **CAC**: $\frac{\text{Acquisition Spend}}{\text{New Customers}}$
- **LTV**: $\text{ARPU} \times \text{Customer Lifespan}$
- **Churn Rate**: $\frac{\text{Customers Lost}}{\text{Total Customers}}$
- **NRR**: $\frac{\text{MRR}_{end} - \text{Churn MRR}}{\text{MRR}_{start}}$
