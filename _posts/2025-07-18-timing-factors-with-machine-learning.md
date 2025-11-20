---
layout: post
title: Timing Factors Using Machine Learning
image: "/posts/factor-title-img.png"
tags: [Finance, Factor Models, Machine Learning, Asset Management, Python]
---

Factor timing has been one of the most debated topics in quantitative finance. Some firms, such as Research Affiliates, argue that factors become “cheap” or “expensive,” making it possible to buy low and sell high. Others, like AQR, find that factor timing is deceptively difficult and rarely outperforms holding static exposures.

In this project, I evaluate whether timing the Value factor (HML) is possible using both traditional econometric models and Machine Learning. I compare multiple timing signals, test out-of-sample performance, incorporate trading costs, and assess whether ML models can meaningfully improve predictive power.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Timing Strategy Overview](#timing-overview)
- [03. OLS Timing Models](#ols-title)
- [04. Drawdown & Sharpe Ratio Results](#drawdowns-title)
- [05. Machine Learning Timing Models](#ml-title)
    - [Logistic Regression](#ml-logistic)
    - [Random Forest](#ml-rf)
    - [Gradient Boosting](#ml-gb)
- [06. ML Model Summary](#ml-summary)
- [07. Trading Costs & Capacity](#trading-title)
- [08. Growth & Next Steps](#growth-next-steps)

___

# Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The core purpose of this project is to test whether the Value factor (HML) can be *timed*—that is, whether exposure should be increased or decreased depending on market conditions. While some researchers claim that value becomes cheap or expensive relative to fundamentals, others argue that factor timing performs no better than market timing.

To investigate this, I evaluated three timing signals, built rolling-window regression strategies, and tested three Machine Learning classifiers to see whether nonlinear patterns could improve forecasting power.

<br>

### Actions <a name="overview-actions"></a>

I constructed a comprehensive factor-timing pipeline that included:

* Three timing signals:
  * **tbl_change** — monthly change in short-term interest rates  
  * **HML_rolling_12m** — trailing 12-month HML performance  
  * **bm_percentile** — book-to-market valuation percentile  

* Rolling 60-month regressions to generate timing signals  
* Out-of-sample performance testing using  
  * Sharpe ratio  
  * Maximum drawdown  
  * KS tests  
  * Out-of-sample \( R^2_{OS} \)

* ML models to predict whether HML would outperform its median:
  * Logistic Regression  
  * Random Forest  
  * Gradient Boosting  

* Transaction-cost simulation at 10–50 bps  
* Analysis of strategy capacity and turnover behavior  

<br>

### Results <a name="overview-results"></a>

Across all models, timing the Value factor proved extremely difficult.

<br>
**Key Findings**

* **bm_percentile** reduced maximum drawdown the most (41.7% → 29.5%).  
* None of the timing signals improved Sharpe ratio vs a static HML exposure.  
* Out-of-sample \( R^2_{OS} \) values were extremely low across signals.  
* ML models detected some nonlinear structure, but not enough to improve forecasting materially.  
* Transaction costs quickly eliminated any potential gains.

<br>
**Machine Learning Result Highlights**

* Logistic Regression: \( R^2_{OS} = 0.0009 \)  
* Random Forest: \( R^2_{OS} = 0.074 \), Sharpe ≈ 0.38  
* Gradient Boosting: \( R^2_{OS} = -0.022 \), Sharpe ≈ 0.37  

Random Forest provided the strongest performance but did not outperform a static HML exposure in a statistically significant way.

<br>

### Growth/Next Steps <a name="overview-growth"></a>

To extend this project further, I plan to:

* Add macroeconomic regime indicators (yield curve slope, inflation surprises)  
* Use deep learning (LSTM, transformers) for sequence-aware forecasting  
* Build long–short multifactor timing instead of single-factor timing  
* Introduce portfolio optimization with turnover and capacity constraints  
* Test cross-sectional factor timing rather than time-series timing  

___

# Data Overview <a name="data-overview"></a>

The dataset includes:

* Monthly HML returns  
* Treasury bill yield changes (tbl_change)  
* Book-to-market percentile ranks  
* Rolling 12-month factor performance  
* Risk-free rates  

These variables were merged into a unified monthly dataset for backtesting regressions and ML classifiers.

___

# Timing Strategy Overview <a name="timing-overview"></a>

To simulate a real asset-management environment, each timing strategy is evaluated out-of-sample using a rolling approach:

1. Fit an OLS regression on the prior 60 months  
2. Extract the coefficient for the timing variable  
3. If coefficient > 0 → invest in HML (signal = 1)  
4. Else → move to cash (signal = 0)  
5. Apply trading costs whenever signals change  
6. Compute cumulative returns and risk metrics  

This framework ensures that only past information is used at each step—no look-ahead bias.

___

# OLS Timing Models <a name="ols-title"></a>

The three timing variables tested were:

* **tbl_change**  
* **HML_rolling_12m**  
* **bm_percentile**

After running rolling regressions, the strategy switches between HML and cash based on the sign of the estimated coefficient.

<br>

### Key Performance Outcomes

* **bm_percentile** reduced drawdown meaningfully  
* **HML_rolling_12m** offered no improvement vs static exposure  
* **tbl_change** produced mild downside protection  
* None improved risk-adjusted returns  

___

# Drawdown & Sharpe Ratio Results <a name="drawdowns-title"></a>

| Strategy | Max Drawdown | Sharpe Ratio |
|---------|--------------|--------------|
| Static HML | **41.73%** | -1.3667 |
| tbl_change | 32.72% | -1.3674 |
| HML_rolling_12m | 41.73% | -1.3667 |
| bm_percentile | **29.48%** | -1.3700 |
| Combined | 41.73% | -1.3667 |

**Interpretation:** timing provided *risk reduction* but not *return improvement*. Sharpe ratios remained almost unchanged or slightly worse.

___

# Machine Learning Timing Models <a name="ml-title"></a>

Below are the ML models used to predict whether next month’s HML return would exceed its historical median (classification target: 1 or 0).

## Logistic Regression <a name="ml-logistic"></a>

**Metrics**

* Out-of-Sample R²: **0.0009**  
* Sharpe Ratio: **0.15**  
* Performance deteriorates at 20–50 bps trading costs  

Although simple and interpretable, Logistic Regression captured almost no timing ability.

<br>

## Random Forest <a name="ml-rf"></a>

**Metrics**

* ROC AUC: **0.652**  
* Out-of-Sample R²: **0.074**  
* Sharpe Ratio: **0.38**  
* Max Drawdown: **26%**

Random Forest captured modest nonlinear relationships and improved cumulative returns but lacked strong statistical significance.

<br>

## Gradient Boosting <a name="ml-gb"></a>

**Metrics**

* ROC AUC: **0.546**  
* Out-of-Sample R²: **-0.022** (overfitting)  
* Sharpe Ratio: **0.37**  
* Max Drawdown: **15.9%**

Gradient Boosting improved downside protection but failed to generalize well out-of-sample.

___

# ML Model Summary <a name="ml-summary"></a>

| Model | Out-of-Sample R² | Sharpe | Max Drawdown |
|-------|------------------|--------|--------------|
| Logistic Regression | 0.0009 | 0.15 | 33%+ |
| Random Forest | **0.074** | **0.38** | 26% |
| Gradient Boosting | -0.022 | 0.37 | **15.9%** |

Random Forest delivered the strongest overall performance but did not demonstrate robust predictive ability relative to a static HML strategy.

___

# Trading Costs & Capacity <a name="trading-title"></a>

To match realistic investment conditions, trading costs were applied whenever the signal switched between “invest” and “no-invest.”

**Results**

* At 10 bps: limited degradation  
* At 20–50 bps: almost all timing benefits disappear  
* Strategy capacity decreases as turnover increases  

This reflects real quant-asset-management constraints: high turnover erodes alpha quickly.

___

# Growth & Next Steps <a name="growth-next-steps"></a>

There are several ways to extend this project:

* Add macroeconomic regime-switching models  
* Use sequence models (LSTM/Transformers) for time-dependent forecasting  
* Explore multi-factor timing signals  
* Introduce Bayesian shrinkage or hierarchical modeling  
* Optimize turnover explicitly using convex or mixed-integer programming  

___
