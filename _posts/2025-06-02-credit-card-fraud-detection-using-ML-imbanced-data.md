---
layout: post
title: Credit Card Fraud Detection Using XGBoost and Cost-Based Threshold Optimization
image: "/posts/fraud-detection-cover.png"
tags: [Fraud Detection, Imbalanced Data, XGBoost, SHAP, Cost Analysis, Python]
---

In this project, we build a credit card fraud detection system using supervised machine learning. We explore the tradeoffs between precision and recall using real-world cost simulations and apply model explainability techniques to ensure interpretability — a key requirement in banking environments like PayPal or Visa.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Concept Overview](#concept-overview)
- [02. Data Overview & Preparation](#data-overview)
- [03. Building and Evaluating Models](#modeling-evaluation)
- [04. Threshold Optimization & Cost Simulation](#threshold-cost)
- [05. Explaining Model Predictions with SHAP](#shap-explainability)
- [06. Final Discussion & Industry Considerations](#discussion)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Financial institutions like PayPal and Visa process millions of transactions daily. Detecting fraud is critical to avoid financial loss, customer dissatisfaction, and regulatory penalties. However, fraud is rare — typically <0.2% — making this a classic imbalanced classification challenge.

This project aims to:
- Detect fraudulent transactions using historical data
- Balance false positives (customer annoyance) and false negatives (missed fraud)
- Ensure decisions are explainable and compliant

<br>

### Actions <a name="overview-actions"></a>

- Conducted EDA on >1.9 million transactions, identifying key fraud patterns
- Engineered features and handled class imbalance
- Trained multiple classifiers (Logistic Regression, Random Forest, XGBoost)
- Tuned classification thresholds based on a custom cost function
- Used SHAP for explainability and feature importance
- Recommended a deployable strategy with real-world considerations

<br>

### Results & Discussion <a name="overview-results"></a>

- XGBoost outperformed other models on F1-score and recall
- Threshold = 0.35 provided best tradeoff (recall = 0.99, precision = 0.44)
- Estimated lowest business cost: **$27,335**
- SHAP showed alignment with domain intuition (balance drops, amount, type)

___

# Concept Overview <a name="concept-overview"></a>

Fraud detection is a **binary classification** problem where the positive class (fraud) is extremely rare. Typical evaluation metrics like accuracy are misleading; instead, we focus on:
- **Recall** (fraud caught / actual fraud)
- **Precision** (fraud caught / flagged as fraud)
- **F1-score** (harmonic mean of precision and recall)

We also introduce a **cost-based framework**, assigning dollar values to:
- False positives: $5 (inconvenience, reputation risk)
- False negatives: $500 (missed fraud, financial loss)

___

# Data Overview & Preparation <a name="data-overview"></a>

### Dataset Summary

- Total transactions: 1,908,786
- Fraud cases: 2,464
- Imbalance ratio: 0.13%

### EDA Insights

- `TRANSFER` and `CASH_OUT` account for most fraud cases
- Fraudulent transactions tend to:
    - Have high `amount`
    - Reduce `oldbalanceOrg` to near zero
    - Involve destination accounts with low balance activity

We engineered features like `balanceDiffOrg` and used:
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical (`type`)
- A unified `Pipeline` for preprocessing and modeling

___

# Building and Evaluating Models <a name="modeling-evaluation"></a>

We trained:
- `LogisticRegression(class_weight="balanced")`
- `RandomForestClassifier()`
- `XGBoostClassifier(scale_pos_weight=...)`

Evaluation focused on fraud recall and precision. XGBoost was selected for:
- Best F1-score
- Stable performance across thresholds
- Compatibility with SHAP for explainability

---

# Threshold Optimization & Cost Simulation <a name="threshold-cost"></a>

### Business Cost Simulation

We tested thresholds from 0.15 to 0.40 and calculated:

| Threshold | Precision | Recall | F1 | FP | FN | Cost |
|-----------|-----------|--------|----|----|----|------|
| 0.15      | 0.32      | 1.00   | 0.48 | 5203 | 10 | $31,015 |
| 0.20      | 0.36      | 0.99   | 0.53 | 4337 | 15 | $29,185 |
| 0.25      | 0.39      | 0.99   | 0.56 | 3752 | 19 | $28,260 |
| 0.30      | 0.42      | 0.99   | 0.59 | 3365 | 22 | $27,825 |
| **0.35**  | 0.44      | 0.99   | 0.61 | 3067 | 24 | **$27,335** ✅ |
| 0.40      | 0.46      | 0.99   | 0.63 | 2848 | 28 | $28,240 |

### Threshold vs Metrics Plot

![threshold_metrics](path_to_threshold_metrics_plot.png)

Threshold = **0.35** provides the lowest estimated business cost while maintaining high fraud recall.

___

# Explaining Model Predictions with SHAP <a name="shap-explainability"></a>

We used **SHAP (TreeExplainer)** for model interpretability.

### SHAP Summary Plot

![shap_summary](path_to_shap_summary.png)

Top features driving predictions:
- `oldbalanceOrg`: large pre-transaction balances
- `newbalanceOrig`: drops to 0 after transfer
- `amount`: high-value transactions
- `type_CASH_OUT`, `type_TRANSFER`: fraud-prone transaction types

These results match industry understanding of fraudulent behavior.

___

# Final Discussion & Industry Considerations <a name="discussion"></a>

### 📌 Business Integration

- Use threshold 0.35 to score transactions in production
- Route high-risk transactions for human review
- Provide SHAP-based rationales to auditors or compliance teams

### 🔮 Future Enhancements

- Add anomaly detection (Isolation Forest, Autoencoders) for unseen fraud types
- Monitor performance drift over time
- Build risk-banding triage system for better resource allocation

### ✅ Conclusion

This project simulates a real-world fraud detection pipeline — optimizing for financial cost, ensuring compliance via explainability, and prioritizing recall in alignment with banking standards like those at PayPal, Visa, or Capital One.

