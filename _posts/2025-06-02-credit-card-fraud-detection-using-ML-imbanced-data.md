---
layout: post
title: Credit Card Fraud Detection on Imbalanced Data Using XGBoost and Cost-Based Threshold Optimization
image: "/posts/fraud-detection-cover.png"
tags: [Fraud Detection, Machine Learning, Classification, XGBoost, Python]
---

In this project, we build a credit card fraud detection system using supervised machine learning on highly imbalanced data, where fraud cases are rare but costly. We explore the tradeoffs between precision and recall using real-world cost simulations and apply model explainability techniques to ensure interoperability, a key requirement in banking environments.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Why XGBoost](#overview-why-xgboost)
- [01. Data Overview](#data-overview)
- [02. Exploratory Data Analysis](#eda)
- [03. Feature Engineering](#feature-engineering)
- [04. Logistic Regression](#logistic-regression)
- [05. Random Forest](#random-forest)
- [06. XGBoost & Threshold Tuning](#xgboost)
- [07. Explainability with SHAP](#shap)
- [08. Summary & Next Steps](#summary)

___

# Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The dataset contains 6.3 million anonymized bank transactions, out of which only **0.13%** are fraudulent. The transactions include features like type, amount, balances, and sender/receiver IDs. 

The extreme class imbalance poses challenges for standard machine learning models, which often default to predicting the majority class (non-fraud). The cost of missing a fraudulent transaction (false negative) is significantly higher than flagging a legitimate one (false positive). This makes **recall** a critical metric.

### Actions <a name="overview-actions"></a>

- Cleaned and explored the data to understand transaction and fraud distribution
- Engineered new features that help isolate anomalies
- Trained and evaluated three models: Logistic Regression, Random Forest, and XGBoost
- Conducted cost-based threshold tuning on XGBoost predictions
- Used SHAP to understand feature influence and ensure model transparency

### Results <a name="overview-results"></a>

| Model              | Recall | Precision | F1-Score | AUC   |
|-------------------|--------|-----------|----------|-------|
| Logistic Regression | 0.93   | 0.02      | 0.04     | —     |
| Random Forest       | 0.77   | 0.97      | 0.86     | —     |
| **XGBoost**         | **0.99** | 0.49    | 0.66     | 0.96  |

### Why XGBoost? <a name="overview-why-xgboost"></a>

- **High Recall:** Catches 99% of fraud cases, reducing costly false negatives
- **Threshold Tuning:** Custom threshold setting minimizes business costs
- **Interpretability:** SHAP values explain model logic for regulatory compliance
- **Imbalance Handling:** `scale_pos_weight` boosts performance on rare fraud class

___

# Data Overview <a name="data-overview"></a>

- **Total records**: 6.36M
- **Fraud cases**: ~8.2K (0.13%)
- **Transaction types**: PAYMENT, TRANSFER, CASH_OUT, etc.

Features include sender/receiver IDs, amounts, and before/after balances.

___

# Exploratory Data Analysis <a name="eda"></a>

### Class Imbalance

![alt text](/img/posts/Fraud-Rate.png "Fraud Rate Distribution")

Shows the rarity of fraud cases.

### Transaction Types

![alt text](/img/posts/Transaction-Type.png "Transaction Type")

Distribution of transaction types (PAYMENT dominates).

### Fraud Rate by Type

![alt text](/img/posts/Fraudulent Distribution-in-Transfer-Cash_Out.png "Fraud by Type")

Frauds mostly occur in TRANSFER and CASH_OUT.

### Transaction Amounts

![alt text](/img/posts/Distribution-of-Transaction-Amounts-(log scale).png "Log Scaled Amounts")

Long-tailed distribution; most transactions are low-value.

### Amount vs. Fraud (under $50k)

![alt text](/img/posts/Amount-vs-isFraud-(Filtered-under-50k).png "Amount vs. isFraud")

Boxplot reveals fraud isn't confined to high amounts.

### Fraud Trends Over Time

![alt text](/img/posts/Number-of-Fraudulent-Transactions-Overtime.png "Fraud Over Time")

Frauds are not evenly distributed over time.

### Correlation Matrix

![alt text](/img/posts/Correlation-Matrix.png "Correlation Matrix")

`balanceDiffOrg` shows moderate correlation with fraud.

___

# Feature Engineering <a name="feature-engineering"></a>

Derived features to highlight discrepancies:
```python
df['balanceDiffOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
```

Dropped irrelevant or non-informative columns:
- `nameOrig`, `nameDest`: anonymized identifiers
- `isFlaggedFraud`: unused feature with negligible positive cases

___

# Logistic Regression <a name="logistic-regression"></a>

### Pipeline

- `StandardScaler` for numerical features
- `OneHotEncoder` for `type`
- Balanced class weights to account for fraud rarity

### Performance

- **High recall** (0.93) but **very low precision** (0.02)
- Detected most frauds but flagged too many non-frauds

![alt text](/img/posts/confusion-matrixviz.png "Confusion Matrix")

___

# Random Forest <a name="random-forest"></a>

A bagged ensemble model trained with class balancing.

### Performance

- **Precision**: 0.97 (almost no false positives)
- **Recall**: 0.77 (missed ~23% fraud)

Effective in low-risk settings but not aggressive enough for high-stakes detection.

___

# XGBoost & Threshold Tuning <a name="xgboost"></a>

Trained with `scale_pos_weight` to address class imbalance.

### Default Threshold

- **Recall**: 0.99
- **Precision**: 0.49
- **F1-score**: 0.66

### Precision-Recall Curve

![alt text](/img/posts/Precision-Recall-Curve.png "PR Curve")

### Threshold Optimization

![alt text](/img/posts/confusion-matrixviz.png "Confusion Matrix at Threshold 0.35")

### Business Cost Simulation

| Threshold | FP   | FN  | Cost |
|-----------|------|-----|-------|
| 0.15      | 5203 | 10  | $31,015 |
| 0.30      | 3365 | 22  | $27,825 |
| **0.35**  | 3067 | 24  | **$27,335** |

![alt text](/img/posts/fraud-detection-threshold-cost.png "Cost Curve")

Selected **threshold = 0.35** based on lowest total cost.

___

# Explainability with SHAP <a name="shap"></a>

SHAP identifies influential features.

![alt text](/img/posts/feature-importance.png "SHAP Feature Importance")

Top drivers of fraud prediction:
- Large `balanceDiffOrg`
- Transaction `amount`
- Transaction `type`

These insights help build trust in ML predictions and ensure compliance.

___

# Summary & Next Steps <a name="summary"></a>

 **XGBoost** chosen for best recall and cost optimization  
 **Threshold tuning** reduced business risk  
 **SHAP** provided transparency for stakeholders  

### Next Steps

- Time-based validation (to simulate concept drift)
- Model deployment with human-in-the-loop review
- Live monitoring of fraud detection KPIs

---

This project reflects how cost-aware modeling, interpretability, and recall-optimized classification align with the goals of modern fraud detection in financial institutions.
