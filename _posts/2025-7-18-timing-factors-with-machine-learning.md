---
layout: post
title: Timing Factors Using Machine Learning
image: "/posts/factor-title-img.png"
tags: [Quantitative Finance, Factor Models, Machine Learning, Asset Management, Python]
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

Factor timing is one of the longest-running debates in systematic asset management. The central question is simple but difficult: **can we predict when a factor will outperform?**  
For the Value factor, measured via **HML (High-minus-Low)** from Fama–French, researchers disagree. Some argue factors become “cheap” or “expensive,” while others find that timing signals break down out-of-sample.

This project tests whether **HML (Value)** can be timed using both traditional econometric models and nonlinear machine learning. The analysis follows industry practice: rolling windows, strict out-of-sample evaluation, turnover-adjusted returns, maximum drawdown analysis, and realistic trading cost assumptions.

---

# 00. Project Overview <a name="overview-main"></a>

## Context <a name="overview-context"></a>

**HML (High-minus-Low)** is the Fama–French Value factor. It represents the return difference between high book-to-market stocks (“value”) and low book-to-market stocks (“growth”).  
While HML has historically delivered a long-term premium, its performance fluctuates sharply. This leads to the question:  
**Can we predict when exposure to Value should be increased or reduced?**

This problem is linked to several well-known challenges:

- **Low signal-to-noise ratio:** Financial return series are extremely noisy, making prediction fragile.  
- **Structural breaks:** Factor behavior changes over time (tech boom, 2008, 2020).  
- **Overfitting risk:** Timing models often perform well in-sample but collapse out-of-sample.  
- **Trading costs:** Even small turnover can destroy timing premiums.  

Academic literature shows similar difficulty. Welch & Goyal (2008) demonstrate that most return predictors produce **negative out-of-sample R²**, and Cochrane (2011) shows that valuation signals have unstable predictive power across regimes.

This project evaluates:

- Three timing signals  
- 60-month rolling OLS regressions  
- Classification-based ML models  
- Out-of-sample performance  
- Trading-cost-adjusted returns  
- Drawdowns and risk metrics  

---

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

Future directions incorporate research on *Factor Investing*, *Model Selection*, and *Active Investment* modules:

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

