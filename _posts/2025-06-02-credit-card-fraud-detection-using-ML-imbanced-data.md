---
layout: post
title: Fraud Detection on Imbalanced Data Using XGBoost and Cost-Based Threshold Optimization
image: "/posts/fraud-detection-cover.png"
tags: [Fraud Detection, Machine Learning, Classification, XGBoost, Python]
---

In this project, we build a mobile money fraud detection system using supervised machine learning on highly imbalanced data, where fraud cases are rare but costly. We explore the tradeoffs between precision and recall using real-world cost simulations and apply model explainability techniques to ensure interoperability, a key requirement in banking environments.

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

**Context**

I worked on a real-world fraud detection problem using a publicly available dataset containing over 6.3 million financial transactions. The dataset simulates mobile money transfers and includes various transaction types such as CASH_IN, CASH_OUT, PAYMENT, TRANSFER, and DEBIT. Fraudulent activity is labeled via the isFraud column and occurs exclusively in TRANSFER and CASH_OUT transactions. A key challenge was the severe class imbalance; only 0.13% of all transactions were marked as fraudulent.

My goal was to develop a machine learning model that could detect fraud with high recall, minimizing false negatives. In financial contexts, undetected fraudulent activity can lead to significant monetary losses and reputational damage, making recall a priority over precision.

### Actions <a name="overview-actions"></a>

I began by performing exploratory data analysis to understand transaction patterns, distributional imbalances, and fraud concentration. I engineered behavioral features like balanceDiffOrg and balanceDiffDest to capture how account balances changed before and after transactions, enhancing the model’s ability to detect anomalies. Since fraud was only present in TRANSFER and CASH_OUT types, I gave these special attention in the analysis and modeling stages.

To handle the imbalance and build robust models, I trained and compared several classifiers:

- **Logistic Regression** with balanced class weights

- **Random Forest**

- **XGBoost**, with custom tuning of scale_pos_weight and classification threshold

I evaluated each model using recall, precision, confusion matrices, and cost-based threshold simulations. XGBoost emerged as the top performer, achieving 99% recall at a threshold of 0.35, with a manageable number of false positives. I also used SHAP to interpret the model and visualize key drivers of fraud predictions, aligning the solution with financial industry requirements for transparency and explainability.

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

Another relevant column is isFlaggedFraud, which indicates whether the system flagged a transaction as potentially fraudulent. However, in the dataset, only 16 transactions were flagged this way, making isFraud the primary label used for training and evaluation.
___

# Data Overview <a name="data-overview"></a>

The dataset contains 6,362,620 records of anonymized financial transactions for fraud detection. Each row represents a single transaction, with variables describing the type of transaction, the amount, and the balances before and after the transfer. The binary target variable isFraud indicates whether a transaction was fraudulent (1) or not (0). Only about 0.13% of transactions are labeled as fraud, making it a highly imbalanced classification problem.

### Class Imbalance

![alt text](/img/posts/Fraud-Rate.png "Fraud Rate Distribution")

### Transaction Types

While fraud is present only in TRANSFER and CASH_OUT transactions, it is important to examine the distribution of all transaction types in the dataset. Understanding which transaction types are most common helps contextualize fraud occurrence within the broader transaction volume. The following chart displays the total count of each transaction type, providing insight into the dataset's structure and class imbalance.

![alt text](/img/posts/Transaction-Type.png "Transaction Type")

### Fraud Over Time

Although the dataset does not include exact timestamps, the step variable provides a proxy for temporal order. Plotting the number of fraudulent transactions per step reveals the frequency and pattern of fraud attempts over time. This visualization shows that fraud is relatively consistent but exhibits certain spikes—offering potential for modeling time-based fraud detection strategies in the future.

![alt text](/img/posts/Number-of-Fraudulent-Transactions-Overtime.png "Fraud Over Time")

### Suspicious Zero-Balance Behavior

One notable pattern in fraudulent activity is that many TRANSFER and CASH_OUT transactions result in the sender’s balance dropping to zero immediately after the transaction. These cases are especially suspicious when the original balance was non-zero, indicating possible cash-outs by fraudsters. This pattern strongly supports the use of engineered features that capture balance shifts.

How the rows were derived:

```python
zero_after_transfer = df[
    (df['oldbalanceOrg'] > 0) &
    (df['newbalanceOrig'] == 0) &
    (df['type'].isin(['TRANSFER', 'CASH_OUT']))
]
```

### Top Transactors & Fraudulent Senders

To better understand the ecosystem of transactions, I explored the most frequent senders, receivers, and customers associated with fraud. This can help in identifying accounts that may be bots or involved in laundering-like behavior. High-frequency actors—especially those repeatedly involved in fraud—can be flagged for additional scrutiny or modeled as risk factors.

```python
top_senders = df['nameOrig'].value_counts().head(10)
top_receivers = df['nameDest'].value_counts().head(10)
fraud_customers = df[df['isFraud'] == 1]['nameOrig'].value_counts().head(10)
```

# Feature Engineering <a name="feature-engineering"></a>

To prepare the data for modeling, I created engineered features to better capture suspicious behavior. In particular, I derived balanceDiffOrg and balanceDiffDest to quantify how balances changed before and after a transaction. These features are useful for identifying anomalies, especially in TRANSFER and CASH_OUT transaction types—where all fraud cases occur.

Derived features to highlight discrepancies:
```python
df['balanceDiffOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
```

Dropped irrelevant or non-informative columns:
- `nameOrig`, `nameDest`: anonymized identifiers
- `isFlaggedFraud`: unused feature with negligible positive cases

___

### Correlation Matrix

To understand how these engineered features relate to fraud, I visualized their correlation with the target variable.

![alt text](/img/posts/Correlation-Matrix.png "Correlation Matrix")

From the correlation matrix above, we can observe that the engineered feature balanceDiffOrg shows the strongest positive correlation with isFraud (correlation = 0.36). This makes sense, as fraudulent transactions often result in the sender’s balance dropping sharply, making this difference a useful signal. In contrast, most other features—including the original balance values—have weak or near-zero correlations with fraud. The balanceDiffDest feature has only a slight correlation (0.03), indicating it may be less informative. Overall, this analysis highlights the value of carefully engineered features over raw balance fields when detecting fraudulent behavior.

**Code used to generate the modeling dataset:**

```python

# Drop ID columns and unused flag
df_model = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Create engineered features
df_model['balanceDiffOrg'] = df_model['oldbalanceOrg'] - df_model['newbalanceOrig']
df_model['balanceDiffDest'] = df_model['newbalanceDest'] - df_model['oldbalanceDest']

# Define features and target
categorical = ['type']
numerical = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

```

**Final Feature Dictionary:**

| Variable Name     | Type        | Description                                                      |
|-------------------|-------------|------------------------------------------------------------------|
| `type`            | Categorical | Transaction type (e.g., TRANSFER, CASH_OUT, PAYMENT)             |
| `amount`          | Numerical   | Transaction amount                                               |
| `oldbalanceOrg`   | Numerical   | Sender’s balance before the transaction                          |
| `newbalanceOrig`  | Numerical   | Sender’s balance after the transaction                           |
| `oldbalanceDest`  | Numerical   | Receiver’s balance before the transaction                        |
| `newbalanceDest`  | Numerical   | Receiver’s balance after the transaction                         |
| `balanceDiffOrg`  | Engineered  | `oldbalanceOrg − newbalanceOrig`; amount debited from sender     |
| `balanceDiffDest` | Engineered  | `newbalanceDest − oldbalanceDest`; amount credited to receiver   |
| `isFraud`         | Target      | `1` = fraudulent transaction, `0` = legitimate transaction        |

___

# Modelling Overview

The goal of this project is to build a classification model that can accurately detect fraudulent financial transactions based on engineered features such as transaction type, transaction amount, and balance shifts before and after the transaction.

Given the severe class imbalance—fraudulent cases represent only 0.13% of the dataset—the primary focus was to maximize recall, minimizing missed fraud cases (false negatives), which are costlier than false alarms.

A machine learning pipeline was implemented to ensure consistency and scalability across all models. This pipeline included:

- Preprocessing using StandardScaler for numerical features and OneHotEncoder for the categorical type variable

- Train-test splitting with stratification on the target variable to maintain class distribution

- Model training and evaluation using cross-validation and business-focused performance metrics

As we are predicting a binary outcome (isFraud), we evaluated the following classification algorithms:

- Logistic Regression with balanced class weights

- Random Forest Classifier

- XGBoost Classifier, tuned with scale_pos_weight and threshold optimization

Each model was assessed using precision, recall, F1-score, and confusion matrices, with XGBoost emerging as the top performer in identifying fraudulent transactions with high recall and interpretable results.

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
