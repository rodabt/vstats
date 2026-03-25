# Growth & Product Analytics Tutorial

A comprehensive guide to using the VStats growth module for product and marketing analytics.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Key Metrics](#key-metrics)
3. [Funnel Analysis](#funnel-analysis)
4. [Cohort Analysis](#cohort-analysis)
5. [Marketing Attribution](#marketing-attribution)
6. [Complete Example](#complete-example)

---

## Getting Started

### Installation

```bash
v install https://github.com/rodabt/vstats
```

### Basic Import

```v
import vstats.growth
```

---

## Key Metrics

The growth module provides all standard product and growth metrics.

### Revenue Metrics

```v
import vstats.growth

fn main() {
    // Average Revenue Per Account (ARPA)
    arpa := growth.arpa(100000.0, 100)
    println("ARPA: ${arpa}")  // $1,000

    // Average Revenue Per User (ARPU)
    arpu := growth.arpu(100000.0, 500)
    println("ARPU: ${arpu}")  // $200

    // Monthly Recurring Revenue (MRR)
    plan_revenues := [50000.0, 30000.0, 20000.0]
    mrr := growth.monthly_recurring_revenue(plan_revenues)
    println("MRR: ${mrr}")  // $100,000

    // Annual Recurring Revenue (ARR)
    arr := growth.annual_recurring_revenue(mrr)
    println("ARR: ${arr}")  // $1,200,000
}
```

### Customer Acquisition

```v
import vstats.growth

fn main() {
    // Customer Acquisition Cost (CAC)
    cac := growth.cac(50000.0, 250)
    println("CAC: ${cac}")  // $200 per customer

    // Payback Period
    payback := growth.payback_period(200.0, 50.0)
    println("Payback: ${payback} months")  // 4 months
}
```

### Lifetime Value

```v
import vstats.growth

fn main() {
    // Simple LTV calculation
    // ARPU: $100, Gross Margin: 80%, Monthly Churn: 5%
    ltv := growth.customer_lifetime_value_simple(100.0, 0.80, 0.05)
    println("LTV: ${ltv}")  // $1,600

    // LTV:CAC Ratio (healthy is 3:1 or higher)
    revenue := 100000.0
    users := 1000
    lifespan := 24.0  // months
    spend := 50000.0
    new_customers := 500

    ltv_cac := growth.ltv_cac_ratio(revenue, users, lifespan, spend, new_customers)
    println("LTV:CAC: ${ltv_cac:.1f}x")

    // Magic Number (SaaS efficiency metric)
    magic := growth.magic_number(1000000.0, 0.70, 500000.0)
    println("Magic Number: ${magic}")  // 1.4 (>1 is good)
}
```

### Retention Metrics

```v
import vstats.growth

fn main() {
    // Churn Rate
    churn := growth.churn_rate(25, 500)
    println("Churn Rate: ${churn * 100:.1f}%")  // 5%

    // Retention Rate
    retention := growth.retention_rate(25, 500)
    println("Retention Rate: ${retention * 100:.1f}%")  // 95%

    // Net Revenue Retention (NRR)
    nrr := growth.net_revenue_retention(100000.0, 115000.0, 5000.0, 20000.0)
    println("NRR: ${nrr * 100:.1f}%")  // 110% (expansion > churn)

    // Gross Revenue Retention (GRR)
    grr := growth.gross_revenue_retention(100000.0, 5000.0)
    println("GRR: ${grr * 100:.1f}%")  // 95%
}
```

### Calculating All Metrics at Once

```v
import vstats.growth

fn main() {
    metrics := growth.calculate_growth_metrics(
        revenue: 500000.0
        accounts: 500
        users: 2500
        acquisition_spend: 100000.0
        new_customers: 500
        customers_lost: 25
        mrr_start: 100000.0
        mrr_end: 115000.0
        churn_mrr: 5000.0
        expansion_mrr: 20000.0
        customer_lifespan: 24.0
    )

    println("ARPA: ${metrics.arpa}")
    println("ARPU: ${metrics.arpu}")
    println("CAC: ${metrics.cac}")
    println("LTV: ${metrics.ltv}")
    println("Churn Rate: ${metrics.churn_rate * 100:.1f}%")
    println("NRR: ${metrics.nrr * 100:.1f}%")
}
```

---

## Funnel Analysis

Funnel analysis tracks user progression through conversion stages.

### Creating a Funnel

```v
import vstats.growth

fn main() {
    // Define funnel stages
    stage_names := ['Visit', 'Signup', 'Activated', 'Paid']
    stage_users := [10000, 2000, 800, 200]

    funnel := growth.create_funnel(stage_names, stage_users)

    println("Overall Conversion: ${funnel.total_conversion * 100:.1f}%")
    println("Drop-off Rate: ${funnel.drop_off_rate * 100:.1f}%")
}
```

### Analyzing Stage Conversions

```v
import vstats.growth

fn main() {
    funnel := growth.create_funnel(
        ['Visit', 'Signup', 'Activated', 'Paid'],
        [10000, 2000, 800, 200]
    )

    // Get detailed conversion data
    conversions := funnel.get_conversions()

    for conv in conversions {
        println("${conv.from_name} -> ${conv.to_name}")
        println("  Conversion Rate: ${conv.rate * 100:.1f}%")
        println("  Drop-off Rate: ${conv.drop_off_rate * 100:.1f}%")
        println("  Users Lost: ${conv.from_users - conv.to_users}")
    }

    // Find biggest drop-off
    worst := funnel.highest_drop_off()
    println("\nBiggest drop-off: ${worst.from_name} -> ${worst.to_name}")

    // Find best conversion
    best := funnel.lowest_drop_off()
    println("Best conversion: ${best.from_name} -> ${best.to_name}")
}
```

### Funnel Leakage Analysis

```v
import vstats.growth

fn main() {
    funnel := growth.create_funnel(
        ['Visit', 'Signup', 'Activated', 'Paid'],
        [10000, 2000, 800, 200]
    )

    // Get users lost at each stage
    leakage := growth.funnel_leakage(funnel)
    println("Leakage by stage:")
    println("  Visit -> Signup: ${leakage[0]} users")
    println("  Signup -> Activated: ${leakage[1]} users")
    println("  Activated -> Paid: ${leakage[2]} users")
}
```

### Segmenting Funnels

```v
import vstats.growth

fn main() {
    // Compare funnels by platform
    segment_data := {
        'Desktop':   [5000, 1500, 600, 150]
        'Mobile':    [5000, 800, 320, 80]
        'Tablet':    [2000, 600, 240, 60]
    }

    segment_funnels := growth.segment_funnel(segment_data)

    for name, funnel in segment_funnels {
        println("${name}: ${funnel.total_conversion * 100:.1f}% conversion")
    }
}
```

### Projecting Conversions

```v
import vstats.growth

fn main() {
    funnel := growth.create_funnel(
        ['Visit', 'Signup', 'Activated', 'Paid'],
        [10000, 2000, 800, 200]
    )

    // Project conversions with 50% more traffic
    projections := growth.projected_conversions(funnel, 5000)
    println("Projected users at each stage:")
    println("  Visit: ${projections[0]}")
    println("  Signup: ${projections[1]}")
    println("  Activated: ${projections[2]}")
    println("  Paid: ${projections[3]}")
}
```

---

## Cohort Analysis

Cohort analysis groups users by acquisition time to track retention over time.

### Creating Cohort Analysis

```v
import vstats.growth

fn main() {
    // Define cohorts by month
    cohort_names := ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024']
    initial_sizes := [1000, 1100, 1200, 1300]

    // Retention data: rows = cohorts, cols = months
    // Month 0, Month 1, Month 2, Month 3
    retention_data := [
        [1000, 800, 650, 550],   // Jan cohort
        [1100, 900, 720, 0],     // Feb cohort (no month 3 data yet)
        [1200, 960, 0, 0],       // Mar cohort
        [1300, 0, 0, 0],         // Apr cohort
    ]

    ca := growth.create_cohort_analysis(cohort_names, initial_sizes, retention_data)

    // Access retention at specific point
    jan_retention_m3 := ca.retention_at_period(0, 3)
    println("Jan cohort, Month 3 retention: ${jan_retention_m3 * 100:.1f}%")

    // Average retention at each month
    for i, avg_ret in ca.avg_retention {
        println("Month ${i} avg retention: ${avg_ret * 100:.1f}%")
    }
}
```

### Adding Revenue Data

```v
import vstats.growth

fn main() {
    mut ca := growth.create_cohort_analysis(
        ['Jan', 'Feb', 'Mar'],
        [100, 110, 120],
        [
            [100, 80, 60],
            [110, 88, 0],
            [120, 96, 0],
        ]
    )

    // Add revenue per cohort per month
    revenue_data := [
        [0.0, 5000.0, 4500.0],
        [0.0, 5500.0, 0.0],
        [0.0, 6000.0, 0.0],
    ]
    ca.set_cohort_revenue(revenue_data)

    // Analyze LTV by period
    for i, ltv in ca.ltv_by_period {
        println("Cumulative LTV through Month ${i}: $${ltv:.2f}")
    }
}
```

### Cohort Comparisons

```v
import vstats.growth

fn main() {
    ca := growth.create_cohort_analysis(
        ['Jan', 'Feb', 'Mar', 'Apr'],
        [100, 100, 100, 100],
        [
            [100, 80, 65, 55],
            [100, 75, 60, 50],
            [100, 82, 68, 58],
            [100, 85, 72, 0],
        ]
    )

    // Compare two cohorts
    jan_ret, mar_ret := ca.compare_cohorts('Jan', 'Mar')
    println("Jan cohort final retention: ${jan_ret * 100:.1f}%")
    println("Mar cohort final retention: ${mar_ret * 100:.1f}%")

    if mar_ret > jan_ret {
        println("Mar cohort is performing better!")
    }
}
```

### Churn Analysis

```v
import vstats.growth

fn main() {
    ca := growth.create_cohort_analysis(
        ['Jan', 'Feb', 'Mar'],
        [100, 100, 100],
        [
            [100, 80, 65],
            [100, 78, 62],
            [100, 82, 68],
        ]
    )

    // Monthly churn rates
    churn_rates := ca.churn_by_period()
    for i, churn in churn_rates {
        println("Month ${i} -> ${i+1} churn: ${churn * 100:.1f}%")
    }
}
```

---

## Marketing Attribution

Attribution models credit marketing channels for conversions.

### First Touch Attribution

Gives 100% credit to the first channel a customer interacts with:

```v
import vstats.growth

fn main() {
    customer_channels := [
        'Google Ads', 'Facebook', 'Google Ads',
        'Organic', 'Email', 'Referral',
        'Google Ads', 'Facebook'
    ]
    conversions := [true, true, true, true, true, true, true, false]
    revenue := [100.0, 150.0, 120.0, 200.0, 80.0, 180.0, 110.0, 0.0]

    results := growth.first_touch_attributes(customer_channels, conversions, revenue)

    for r in results {
        println("${r.channel}: $${r.revenue:.2f} (${r.conversions} conversions)")
        println("  Attribution: ${r.attribution_score * 100:.1f}%")
    }
}
```

### Last Touch Attribution

Gives 100% credit to the last channel before conversion:

```v
import vstats.growth

fn main() {
    customer_channels := ['Google', 'Facebook', 'Email', 'Organic']
    conversions := [true, true, true, true]
    revenue := [100.0, 200.0, 150.0, 175.0]

    results := growth.last_touch_attributes(customer_channels, conversions, revenue)

    for r in results {
        println("${r.channel}: $${r.revenue:.2f}")
    }
}
```

### Linear Attribution

Gives equal credit to all touchpoints:

```v
import vstats.growth

fn main() {
    customer_touchpoints := [
        ['Google', 'Facebook', 'Email'],
        ['Organic', 'Google', 'Facebook'],
        ['Email', 'Referral']
    ]
    conversions := [true, true, true]
    revenue := [300.0, 400.0, 200.0]

    results := growth.linear_attributes(customer_touchpoints, conversions, revenue)

    for r in results {
        println("${r.channel}: $${r.revenue:.2f}")
    }
}
```

### Time Decay Attribution

Gives more credit to recent touchpoints using exponential decay:

```v
import vstats.growth

fn main() {
    customer_touchpoints := [
        ['Google', 'Facebook', 'Email'],
        ['Organic', 'Google']
    ]
    touchpoint_days := [
        [14, 7, 1],   // Days before conversion
        [21, 1]
    ]
    conversions := [true, true]
    revenue := [300.0, 400.0]

    // 7-day half-life: touchpoints closer to conversion get more credit
    results := growth.time_decay_attributes(customer_touchpoints, touchpoint_days, conversions, revenue, 7.0)

    for r in results {
        println("${r.channel}: $${r.revenue:.2f}")
    }
}
```

### Position Based Attribution

Gives 40% to first, 40% to last, 20% distributed among middle touchpoints:

```v
import vstats.growth

fn main() {
    customer_touchpoints := [
        ['Google', 'Facebook', 'Email', 'Sales'],
        ['Organic', 'Google']
    ]
    conversions := [true, true]
    revenue := [500.0, 300.0]

    results := growth.position_based_attributes(customer_touchpoints, conversions, revenue)

    for r in results {
        println("${r.channel}: $${r.revenue:.2f}")
    }
}
```

### Channel ROI

Calculate return on investment for each channel:

```v
import vstats.growth

fn main() {
    results := [
        growth.AttributionResult{
            channel: 'Google Ads'
            conversions: 50
            revenue: 5000.0
            attribution_score: 0.4
        },
        growth.AttributionResult{
            channel: 'Facebook'
            conversions: 30
            revenue: 3000.0
            attribution_score: 0.24
        },
        growth.AttributionResult{
            channel: 'Email'
            conversions: 20
            revenue: 2000.0
            attribution_score: 0.16
        },
    ]

    channel_costs := {
        'Google Ads': 1000.0
        'Facebook':   600.0
        'Email':       200.0
    }

    roi := growth.channel_roi(results, channel_costs)

    println("Channel ROI:")
    for channel, value in roi {
        println("  ${channel}: ${value * 100:.1f}%")
    }
}
```

### ROAS (Return on Ad Spend)

```v
import vstats.growth

fn main() {
    // Individual channel ROAS
    google_roas := growth.roas(5000.0, 1000.0)
    println("Google ROAS: ${google_roas:.1f}x")

    // Blended ROAS across all channels
    blended := growth.blended_roas(15000.0, 3500.0)
    println("Blended ROAS: ${blended:.1f}x")
}
```

---

## Complete Example

End-to-end growth analytics workflow:

```v
module main

import vstats.growth

fn main() {
    println("=== Growth Analytics Dashboard ===\n")

    // 1. Revenue Metrics
    println("1. Revenue Overview")
    mrr := growth.monthly_recurring_revenue([75000.0, 25000.0, 15000.0])
    arr := growth.annual_recurring_revenue(mrr)
    println("   MRR: $${mrr:,.0f}")
    println("   ARR: $${arr:,.0f}")

    // 2. Customer Metrics
    println("\n2. Customer Metrics")
    cac := growth.cac(50000.0, 200)
    println("   CAC: $${cac:.0f}")

    ltv := growth.customer_lifetime_value_simple(125.0, 0.75, 0.05)
    println("   LTV: $${ltv:.0f}")
    println("   LTV:CAC: ${(ltv/cac):.1f}x")

    payback := growth.payback_period(cac, 125.0)
    println("   Payback Period: ${payback:.1f} months")

    // 3. Retention
    println("\n3. Retention Metrics")
    churn := growth.churn_rate(15, 500)
    retention := growth.retention_rate(15, 500)
    println("   Churn Rate: ${churn * 100:.1f}%")
    println("   Retention Rate: ${retention * 100:.1f}%")

    nrr := growth.net_revenue_retention(100000.0, 112000.0, 5000.0, 17000.0)
    println("   NRR: ${nrr * 100:.1f}%")

    // 4. Funnel Analysis
    println("\n4. Conversion Funnel")
    funnel := growth.create_funnel(
        ['Visitor', 'Trial Signup', 'Activated', 'Paid'],
        [25000, 5000, 2500, 500]
    )
    println("   Overall Conversion: ${funnel.total_conversion * 100:.2f}%")

    conversions := funnel.get_conversions()
    for conv in conversions {
        println("   ${conv.from_name} -> ${conv.to_name}: ${conv.rate * 100:.1f}%")
    }

    worst := funnel.highest_drop_off()
    println("   Worst Stage: ${worst.from_name} -> ${worst.to_name}")

    // 5. Cohort Analysis
    println("\n5. Cohort Retention")
    ca := growth.create_cohort_analysis(
        ['Jan', 'Feb', 'Mar', 'Apr'],
        [500, 550, 600, 650],
        [
            [500, 400, 340, 290],
            [550, 440, 374, 0],
            [600, 492, 418, 0],
            [650, 533, 0, 0],
        ]
    )

    for i, avg in ca.avg_retention {
        println("   Month ${i}: ${avg * 100:.1f}% retention")
    }

    // 6. Marketing Attribution
    println("\n6. Channel Attribution")
    results := growth.first_touch_attributes(
        ['Google', 'Facebook', 'Google', 'Organic', 'Email', 'Google'],
        [true, true, true, true, true, true],
        [500.0, 300.0, 450.0, 600.0, 250.0, 400.0]
    )

    total_revenue := results.sum[f64](0.0, fn(acc f64, r growth.AttributionResult) f64 { return acc + r.revenue })

    for r in results {
        pct := if total_revenue > 0 { r.revenue / total_revenue * 100 } else { 0.0 }
        println("   ${r.channel}: $${r.revenue:,.0f} (${pct:.1f}%)")
    }

    // Channel ROI
    channel_costs := {
        'Google':   800.0
        'Facebook': 400.0
        'Organic':  200.0
        'Email':    100.0
    }
    roi := growth.channel_roi(results, channel_costs)

    println("\n   Channel ROI:")
    for channel, value in roi {
        println("   ${channel}: ${value * 100:.1f}%")
    }

    println("\n=== Dashboard Complete ===")
}
```

---

## Key Formulas

| Metric | Formula |
|--------|---------|
| ARPA | Revenue / Accounts |
| ARPU | Revenue / Users |
| CAC | Acquisition Spend / New Customers |
| LTV | ARPU × Customer Lifespan |
| Churn Rate | Customers Lost / Total Customers |
| Retention Rate | 1 - Churn Rate |
| NRR | (MRR_end - Churn_MRR) / MRR_start |
| GRR | (MRR_start - Churn_MRR) / MRR_start |
| Payback Period | CAC / Monthly ARPU |
| Magic Number | (Net New ARR × Gross Margin) / S&M Spend |
| ROAS | Revenue / Ad Spend |

---

## Best Practices

1. **Track cohorts monthly** - Group users by acquisition month for retention analysis
2. **Use NRR over GRR** - NRR accounts for expansion revenue
3. **Target 3:1 LTV:CAC** - Industry standard for healthy unit economics
4. **Combine attribution models** - Use multiple models for different perspectives
5. **Monitor funnel drop-offs** - Focus optimization efforts on biggest leaks
