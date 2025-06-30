
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

The dataset contains 6.3 million anonymized bank transactions, out of which only **0.13%** are fraudulent. The transactions include features like type, amount, balances, and IDs.

### Actions <a name="overview-actions"></a>

- Conducted EDA to analyze transaction patterns
- Engineered balance-difference features
- Built pipelines for Logistic Regression, Random Forest, and XGBoost
- Prioritized recall and used threshold tuning to optimize business impact
- Used SHAP for model interpretability

### Results <a name="overview-results"></a>

| Model              | Recall | Precision | F1-Score | AUC   |
|-------------------|--------|-----------|----------|-------|
| Logistic Regression | 0.93   | 0.02      | 0.04     | —     |
| Random Forest       | 0.77   | 0.97      | 0.86     | —     |
| **XGBoost**         | **0.99** | 0.49    | 0.66     | 0.96  |

### Why XGBoost? <a name="overview-why-xgboost"></a>

- **Best recall (99%)** — critical in fraud detection
- Effective class imbalance handling with `scale_pos_weight`
- Tunable thresholds enabled cost-optimized predictions
- Compatible with SHAP for interpretation

___

# Data Overview <a name="data-overview"></a>

Dataset: 6.3M records, 11 features

Key columns:
- `type`: transaction type (PAYMENT, TRANSFER, CASH_OUT...)
- `amount`, `oldbalanceOrg`, `newbalanceOrig`, etc.
- `isFraud`: target (1 = fraud, 0 = not fraud)

Fraud rate: **0.13%**

___

# Exploratory Data Analysis <a name="eda"></a>

### Class Imbalance

![alt text](/img/posts/Fraud-Rate.png "Fraud Rate Distribution")

### Transaction Types

![alt text](/img/posts/Transaction-Type.png "Transaction Type")

### Fraud Rate by Type

![alt text](/img/posts/Fraudulent Distribution-in-Transfer-Cash_Out.png "Fraud by Type")

TRANSFER and CASH_OUT have the highest fraud rates.

### Transaction Amounts

![alt text](/img/posts/Distribution-of-Transaction-Amounts-(log scale).png "Log Scaled Amounts")

Log scale shows long tail; high-value frauds exist.

### Amount vs. Fraud (under $50k)

![alt text](/img/posts/Amount-vs-isFraud-(Filtered-under-50k).png "Amount vs. isFraud")

Frauds tend to occur across small and mid-range values too.

### Fraud Trends Over Time

![alt text](/img/posts/Number-of-Fraudulent-Transactions-Overtime.png "Fraud Over Time")

### Correlation Matrix

![alt text](/img/posts/Correlation-Matrix.png "Correlation Matrix")

Some features like `balanceDiffOrg` and `balanceDiffDest` showed correlation with `isFraud`.

___

# Feature Engineering <a name="feature-engineering"></a>

Created two new features:

```python
df['balanceDiffOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
```

Dropped `nameOrig`, `nameDest`, and `isFlaggedFraud`.

___

# Logistic Regression <a name="logistic-regression"></a>

Logistic regression was implemented using a pipeline with:
- `StandardScaler` for numeric features
- `OneHotEncoder` for `type`

### Results

- **Recall**: 0.93
- **Precision**: 0.02
- **F1**: 0.04

![alt text](/img/posts/confusion-matrixviz.png "Confusion Matrix")

LogReg caught most frauds, but with many false positives.

___

# Random Forest <a name="random-forest"></a>

Using class-weighted Random Forest pipeline.

### Results

- **Recall**: 0.77
- **Precision**: 0.97
- **F1**: 0.86

This model had **high precision**, but missed too many frauds for our business case.

___

# XGBoost & Threshold Tuning <a name="xgboost"></a>

With `scale_pos_weight` and evaluation via multiple thresholds.

### Default Threshold

- **Recall**: 0.99
- **Precision**: 0.49
- **F1**: 0.66

### Precision-Recall Curve

![alt text](/img/posts/Precision-Recall-Curve.png "PR Curve")

### Confusion Matrix (Threshold = 0.35)

![alt text](/img/posts/confusion-matrixviz.png "Confusion Matrix at Threshold 0.35")

### Threshold Cost Analysis

| Threshold | False Positives | False Negatives | Total Cost |
|-----------|------------------|------------------|-------------|
| 0.15      | 5,203            | 10               | $31,015     |
| 0.30      | 3,365            | 22               | $27,825     |
| **0.35**  | 3,067            | 24               | **$27,335** |

![alt text](/img/posts/fraud-detection-threshold-cost.png "Cost Curve vs Threshold")

___

# Explainability with SHAP <a name="shap"></a>

SHAP shows which features most influence predictions.

![alt text](/img/posts/feature-importance.png "SHAP Feature Importance")

Top features:
- `balanceDiffOrg`
- `balanceDiffDest`
- `amount`
- Encoded transaction types

___

# Summary & Next Steps <a name="summary"></a>

✅ **XGBoost** was chosen for its 99% recall  
✅ **Threshold tuning** minimized business cost  
✅ **SHAP** helped with explainability  

### Next Steps

- Deploy model as an API
- Implement time-based validation
- Monitor for concept drift
- Add human-in-the-loop review flow

---

This project illustrates how to design and deploy a robust, business-aligned fraud detection system. Accuracy alone is not enough — real-world systems need to balance risk, recall, interpretability, and cost.
