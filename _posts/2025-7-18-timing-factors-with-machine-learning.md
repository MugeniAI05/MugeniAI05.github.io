---
layout: post
title: Timing Factors Using Machine Learning
image: "/posts/factor-title-img.png"
tags: [Quantitative Finance, Factor Models, Machine Learning, Asset Management, Python]
---

Factor timing is one of the most debated problems in quantitative finance. Some practitioners, such as Research Affiliates, argue that factors become “cheap” or “expensive,” creating opportunities to buy low and sell high. Others — notably AQR — argue that factor timing is extremely fragile, prone to overfitting, and rarely delivers reliable outperformance out-of-sample.

Academic evidence supports this skepticism.  
Cochrane (2011) shows that valuation ratios predict returns in-sample, but their out-of-sample forecasting power is weak. Welch & Goyal (2008) famously demonstrate that most market-return predictors fail to beat the historical mean. The combination of low signal-to-noise ratios, structural breaks, and high turnover makes timing risk premia exceptionally difficult.

This project tests those claims through a comprehensive empirical analysis of **HML (Value)** timing. Using both traditional rolling-window regressions and machine-learning models, I examine whether any signal can reliably forecast when HML will outperform. All evaluations are fully out-of-sample and adjusted for trading costs, mirroring how systematic portfolio managers vet factor-timing strategies in production environments.

---

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

---

# 00. Project Overview <a name="overview-main"></a>

## Context <a name="overview-context"></a>

The goal of this project is to determine whether the **Value factor (HML)** can be *timed*. In other words:  
**Should we increase or decrease exposure to Value depending on market conditions?**

This question sits at the heart of modern quantitative investing:

* Some argue factors possess **valuation-driven cycles**, which can be exploited.  
* Others find that factors behave like **risk premia**, which are difficult to forecast.

The CMU *Data Analytics in Finance* material emphasizes that timing typically fails when:

- Predictability is weak relative to noise  
- Models rely heavily on in-sample fits  
- Trading costs and turnover overwhelm small edges  
- Structural breaks shift factor behavior  

These risks guide the structure of this project.

I evaluated three economically motivated timing signals, built rolling-window OLS models, and tested whether machine learning can uncover nonlinear predictability patterns. All results are out-of-sample.

<br>

## Actions <a name="overview-actions"></a>

To replicate a professional factor-research process, I constructed a complete timing pipeline:

**Signals Tested**
- `tbl_change`: monthly change in short-term rates  
- `HML_rolling_12m`: momentum-style trailing HML performance  
- `bm_percentile`: valuation percentile of book-to-market  

**Methodology**
- Rolling 60-month OLS timing regressions  
- Out-of-sample evaluation using Sharpe, drawdown, and \( R^2_{OS} \)  
- ML classifiers: Logistic Regression, Random Forest, Gradient Boosting  
- ROC-AUC, KS tests, and classification performance  
- Transaction-cost simulations at 10–50 bps  
- Turnover & capacity analysis  

This aligns with the **universe → signal → model → trading rule → evaluation** structure emphasized in quant AM frameworks.

<br>

## Results <a name="overview-results"></a>

Across econometric and ML approaches, **HML proved extremely difficult to time**.  
This matches published evidence suggesting that factor timing edges are small, unstable, and easily eliminated by costs.

### Key Findings

- **bm_percentile reduced max drawdown** from 41.7% → **29.5%**, consistent with valuation mean reversion.  
- None of the signals improved Sharpe ratio relative to static HML.  
- Out-of-sample \( R^2_{OS} \) values were extremely low across signals.  
- Random Forest captured modest nonlinear structure but lacked robust generalizability.  
- Trading costs erased nearly all incremental benefits at 20–50 bps.

### ML Result Highlights
- Logistic Regression: \( R^2_{OS} = 0.0009 \)  
- Random Forest: \( R^2_{OS} = 0.074 \), Sharpe ≈ **0.38**  
- Gradient Boosting: \( R^2_{OS} = -0.022 \), Sharpe ≈ **0.37**  

Random Forest provided the strongest performance but still failed to outperform the static benchmark with statistical confidence.

<br>

## Growth/Next Steps <a name="overview-growth"></a>

Future directions incorporate lessons from CMU’s *Factor Investing*, *Model Selection*, and *Active Investment* modules:

- Add macroeconomic regime indicators (yield curve, credit spreads, inflation surprises)  
- Use sequence-aware models (LSTM, Transformers)  
- Apply Bayesian shrinkage (e.g., Campbell–Thompson constraints)  
- Explore multi-factor timing instead of single-factor timing  
- Implement turnover-aware portfolio optimization  
- Examine cross-sectional factor timing, which often exhibits stronger signals  

---

# 01. Data Overview <a name="data-overview"></a>

Dataset includes monthly:

- HML factor returns  
- Changes in Treasury bill yields (`tbl_change`)  
- Book-to-market percentile ranks (`bm_percentile`)  
- Rolling 12-month HML performance  
- Risk-free rate  

All variables are merged into a unified monthly dataset for OLS and ML modeling.

---

# 02. Timing Strategy Overview <a name="timing-overview"></a>

Each timing model is evaluated using a realistic **rolling-window** approach:

1. Fit an OLS regression using the *previous 60 months*  
2. Extract the coefficient for the timing signal  
3. If coefficient > 0 ⇒ invest in HML (signal = 1)  
4. Else ⇒ stay in cash (signal = 0)  
5. Apply transaction costs whenever signals switch  
6. Compute cumulative return, Sharpe, and drawdown metrics  

This ensures **no look-ahead bias**, aligning with best practices described under *Backtests* in the CMU Active Investment material.

---

# 03. OLS Timing Models <a name="ols-title"></a>

Signals tested:

- `tbl_change`  
- `HML_rolling_12m`  
- `bm_percentile`  

### Key Outcomes

- `bm_percentile` produced meaningful drawdown improvements  
- `HML_rolling_12m` behaved similarly to static HML  
- `tbl_change` offered mild downside protection  
- None improved Sharpe ratio  

These patterns echo findings that simple timing rules rarely deliver out-of-sample gains (Welch & Goyal, 2008).

---

# 04. Drawdown & Sharpe Ratio Results <a name="drawdowns-title"></a>

| Strategy | Max Drawdown | Sharpe Ratio |
|---------|--------------|--------------|
| Static HML | **41.73%** | -1.3667 |
| tbl_change | 32.72% | -1.3674 |
| HML_rolling_12m | 41.73% | -1.3667 |
| bm_percentile | **29.48%** | -1.3700 |
| Combined | 41.73% | -1.3667 |

**Interpretation:**  
Timing reduced **risk** but did not improve **risk-adjusted returns**.

---

# 05. Machine Learning Timing Models <a name="ml-title"></a>

The ML task is to classify whether *next month’s HML return exceeds its historical median*.

## Logistic Regression <a name="ml-logistic"></a>

**Metrics**
- Out-of-Sample \(R^2\): **0.0009**  
- Sharpe: **0.15**  
- Highly sensitive to cost assumptions  

As expected, the linear model struggled to extract meaningful signals.

## Random Forest <a name="ml-rf"></a>

**Metrics**
- ROC AUC: **0.652**  
- Out-of-Sample \(R^2\): **0.074**  
- Sharpe: **0.38**  
- Max Drawdown: **26%**  

Captured modest nonlinear structure but still lacked strong predictive power.

## Gradient Boosting <a name="ml-gb"></a>

**Metrics**
- ROC AUC: **0.546**  
- Out-of-Sample \(R^2\): **-0.022**  
- Sharpe: **0.37**  
- Max Drawdown: **15.9%**  

Improved downside protection but showed signs of overfitting.

---

# 06. ML Model Summary <a name="ml-summary"></a>

| Model | Out-of-Sample R² | Sharpe | Max Drawdown |
|-------|------------------|--------|--------------|
| Logistic Regression | 0.0009 | 0.15 | 33%+ |
| Random Forest | **0.074** | **0.38** | 26% |
| Gradient Boosting | -0.022 | 0.37 | **15.9%** |

Random Forest performed best, but none demonstrated robust ability to time HML.

---

# 07. Trading Costs & Capacity <a name="trading-title"></a>

Trading costs are applied each time the model switches between “invest” and “cash.”

**Findings**

- 10 bps: minimal impact  
- 20–50 bps: effectively eliminates all timing benefits  
- Higher turnover = lower capacity  

This mirrors the CMU material: turnover is the dominant friction for factor timing.

---

# 08. Growth & Next Steps <a name="growth-next-steps"></a>

Next iterations will:

- Integrate macroeconomic regime-switching  
- Test multi-factor timing signals  
- Use deep learning for sequence dependency  
- Apply shrinkage-based constraints (Campbell–Thompson)  
- Introduce turnover-constrained optimization  
- Explore cross-sectional timing techniques  

These advances aim to improve stability, reduce noise, and align with professional quant-research practices.

---

