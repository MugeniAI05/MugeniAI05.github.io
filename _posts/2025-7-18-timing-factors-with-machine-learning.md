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

The dataset integrates:

- **HML** monthly returns  
- **RF (Risk-Free Rate)** from Fama–French  
- **tbl (Treasury bill yield)** from macro predictors  
- **b/m (book-to-market ratio)**  
- Rolling 12-month HML average  
- Derived signals:
  - tbl_change  
  - HML_rolling_12m  
  - bm_percentile  

All series are merged by month and aligned for backtesting.

```python
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load data
predictor_data = pd.read_csv('PredictorData2019.csv')
ff_factors_data = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv')

# Clean/align dates
predictor_data["DATE"] = pd.to_datetime(predictor_data["DATE"], format="%m/%d/%Y") + pd.offsets.MonthEnd(0)
ff_factors_data.rename(columns={"Unnamed: 0": "DATE"}, inplace=True)
ff_factors_data["DATE"] = pd.to_datetime(ff_factors_data["DATE"], format="%Y%m") + pd.offsets.MonthEnd(0)

# Select relevant columns
HML_data = ff_factors_data[["DATE", "HML", "RF"]]
macro_data = predictor_data[["DATE", "tbl", "b/m"]]

# Merge into a single monthly dataframe
df = pd.merge(HML_data, macro_data, on="DATE", how="inner")
df.set_index("DATE", inplace=True)

# Derived signals
df["tbl_change"] = df["tbl"].diff()
df["HML_rolling_12m"] = df["HML"].rolling(window=12).mean()
df["bm_percentile"] = df["b/m"].rank(pct=True)

# Drop initial NaNs from rolling/diff
df.dropna(inplace=True)

```

---

# 02. Timing Strategy Overview <a name="timing-overview"></a>

Each timing model is evaluated using a realistic **rolling-window** approach:

1. Fit an OLS regression using the *previous 60 months*  
2. Extract the coefficient for the timing signal  
3. If coefficient > 0 ⇒ invest in HML (signal = 1)  
4. Else ⇒ stay in cash (signal = 0)  
5. Apply transaction costs whenever signals switch  
6. Compute cumulative return, Sharpe, and drawdown metrics  

This replicates realistic institutional backtesting and avoids look-ahead bias.

```python
def backtest_timing_strategy(df, timing_signal_column):
    betas, p_values, timing_returns, r_squared = [], [], [], []
    window_size = 60

    for i in range(len(df) - window_size):
        train_data = df.iloc[i : i + window_size]

        X_train = sm.add_constant(train_data[[timing_signal_column]])
        Y_train = train_data["HML"]

        model = sm.OLS(Y_train, X_train).fit()

        betas.append(model.params[timing_signal_column])
        p_values.append(model.pvalues[timing_signal_column])
        r_squared.append(model.rsquared)

        next_month = df.iloc[i + window_size]

        if model.params[timing_signal_column] > 0:
            timing_returns.append(next_month["HML"])
        else:
            timing_returns.append(0)

    timing_cum_returns = [1]
    static_cum_returns = [1]

    for i in range(len(timing_returns)):
        timing_cum_returns.append(
            timing_cum_returns[-1] * (1 + timing_returns[i] / 100)
        )
        static_cum_returns.append(
            static_cum_returns[-1] * (1 + df["HML"].values[i + window_size] / 100)
        )

    timing_cum_returns = np.array(timing_cum_returns[1:])
    static_cum_returns = np.array(static_cum_returns[1:])

    def calculate_max_drawdown(cum_returns):
        peaks = np.maximum.accumulate(cum_returns)
        drawdowns = (peaks - cum_returns) / np.where(peaks == 0, 1e-10, peaks)
        return np.nanmax(drawdowns)

    timing_max_dd = calculate_max_drawdown(timing_cum_returns)
    static_max_dd = calculate_max_drawdown(static_cum_returns)

    rf_values = df["RF"].values[window_size:]
    timing_excess_returns = np.array(timing_returns) / 100 - rf_values
    static_excess_returns = df["HML"].values[window_size:] / 100 - rf_values

    timing_sharpe_ratio = np.mean(timing_excess_returns) / np.std(timing_excess_returns)
    static_sharpe_ratio = np.mean(static_excess_returns) / np.std(static_excess_returns)

    t_stat, p_value = ttest_ind(timing_returns, df["HML"].values[window_size:])

    beta_significance = np.mean(np.array(p_values) < 0.05)

    ks_stat, ks_p_value = ks_2samp(timing_returns, df["HML"].values[window_size:])

    return {
        "T-statistic": t_stat,
        "P-value": p_value,
        "Beta Significance": beta_significance,
        "Sharpe Ratio (Timing)": timing_sharpe_ratio,
        "Sharpe Ratio (Static)": static_sharpe_ratio,
        "Max Drawdown (Timing)": timing_max_dd,
        "Max Drawdown (Static)": static_max_dd,
        "Kolmogorov-Smirnov Statistic": ks_stat,
        "Kolmogorov-Smirnov P-value": ks_p_value
    }
```
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

```python
timing_results = {}

# tbl_change
timing_results["tbl_change"] = backtest_timing_strategy(df, "tbl_change")

# HML_rolling_12m
timing_results["HML_rolling_12m"] = backtest_timing_strategy(df, "HML_rolling_12m")

# bm_percentile
timing_results["bm_percentile"] = backtest_timing_strategy(df, "bm_percentile")

# combined
df["combined_signal"] = df["tbl_change"] + df["HML_rolling_12m"] + df["bm_percentile"]
timing_results["combined"] = backtest_timing_strategy(df, "combined_signal")

timing_results_df = pd.DataFrame.from_dict(timing_results, orient="index")
timing_results_df

```
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

