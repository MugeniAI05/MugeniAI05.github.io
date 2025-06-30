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
    - [Growth/Next Steps](#overview-next-steps)
    - [Key Definitions](#key-definitions)
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

I worked on a real-world fraud detection problem using a financial transactions dataset containing over 6 million records. A major challenge in this dataset was severe class imbalance; fraudulent transactions made up only 0.13% of all entries.

My objective was to build a machine learning model that could detect fraud with high recall, since missing fraudulent activity (false negatives) is far more costly than flagging legitimate transactions (false positives) in the banking industry.

### Actions <a name="overview-actions"></a>

I started by exploring the dataset to identify transaction types, detect patterns, and engineer meaningful features like balanceDiffOrg and balanceDiffDest. I focused on high-risk transaction types such as TRANSFER and CASH_OUT, where most fraud cases occurred.

To address the class imbalance, I tested several classification models:

- **Logistic Regression** with balanced class weights

- **Random Forest**

- **XGBoost**, tuned with scale_pos_weight and custom threshold adjustments

I evaluated performance using precision, recall, confusion matrices, and business cost simulations. XGBoost emerged as the top performer, achieving 99% recall at a 0.35 classification threshold, with manageable false positives. I used SHAP values to interpret model predictions, ensuring explainability for a finance context.

### Results <a name="overview-results"></a>

My testing showed that XGBoost achieved the best fraud detection performance in terms of recall, which was the primary metric of interest due to the high cost of false negatives in a financial context.

**Metric 1: Classification Report (Test Set, Default Threshold)**

| Model              | Recall | Precision | F1-Score | AUC   |
|-------------------|--------|-----------|----------|-------|
| Logistic Regression | 0.93   | 0.02      | 0.04     | —     |
| Random Forest       | 0.77   | 0.97      | 0.86     | —     |
| **XGBoost**         | **0.99** | 0.49    | 0.66     | 0.96  |


**Metric 2: Confusion Matrix at Optimized Threshold (0.35)**

![alt text](/img/posts/confusion-matrixviz.png "Confusion Matrix")

**Metric 3: Business Cost Simulation**

Using assumed costs of:

- False Positive (FP) = $5 (customer inconvenience)

- False Negative (FN) = $500 (financial loss)

| Threshold           | FP | FN | Total Cost |
|-------------------|--------|-----------|----------|
| 0.15       | 5,203 | 10     | $31,015     |
| 0.30        | 3,365 | 22      | $27,825    |
| **0.35 **      | **3,067** | **24**    | **$27,335** (lowest)     |
| 0.40       | 2,848  | 0.49    | $28,240     |

Based on this cost simulation, I selected 0.35 as the optimal decision threshold for deployment.

**Model Explainability**

To ensure transparency, I used SHAP to analyze the XGBoost model:

The top features influencing fraud predictions were:

- amount,

- balanceDiffOrg,

- balanceDiffDest,

- and encoded transaction type.

SHAP summary plots helped visualize feature importance, supporting financial compliance and auditability.

![alt text](/img/posts/feature-importance.png "SHAP Feature Importance")

### Growth/Next Steps <a name="overview-next-steps"></a>

While the XGBoost model delivered high recall, particularly important in minimizing missed fraudulent transactions, further improvements could be made to enhance overall performance and production readiness. Exploring additional modeling techniques such as LightGBM or CatBoost may improve precision without compromising recall. Incorporating time-based validation strategies would help detect concept drift, ensuring the model adapts as fraud tactics evolve. Periodic retraining—monthly or quarterly—could help maintain model effectiveness in a changing environment.

From a deployment perspective, setting up model monitoring would be critical. Tracking metrics such as daily fraud detection rates, recall, and false positive volume would enable early detection of performance issues. A human-in-the-loop system could be implemented to review high-risk predictions, ensuring a balance between customer experience and fraud prevention. Additionally, enriching the dataset with behavioral signals, device metadata, and engineered features like transaction frequency and velocity would likely strengthen the model’s ability to flag anomalous patterns.

### Key Definitions <a name="key-definitions"></a>

In this project, fraud is defined according to the dataset’s isFraud column, where a value of 1 indicates a fraudulent transaction and 0 indicates a legitimate one. Fraudulent transactions occur exclusively in TRANSFER and CASH_OUT transaction types. These involve money being moved from one account to another (TRANSFER) or withdrawn from an account (CASH_OUT).

Another relevant column is isFlaggedFraud, which indicates whether a transaction was flagged by the system as potentially fraudulent. However, in the dataset, only 16 transactions were flagged this way, making isFraud the primary label used for training and evaluation.
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
