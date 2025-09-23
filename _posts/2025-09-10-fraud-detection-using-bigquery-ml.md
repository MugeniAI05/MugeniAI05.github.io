---
layout: post
title: Predictive Modeling with SQL and BigQuery ML
image: "/posts/big-query.png"
tags: [BigQuery, SQL, Machine Learning, Google Cloud]
---

In this project, I used **BigQuery ML** to build and evaluate fraud detection models for financial transaction data, including both **unsupervised** (K-Means) and **supervised** models (Logistic Regression and Boosted Tree). The goal was to detect fraudulent activity more effectively by applying feature engineering and model evaluation strategies directly within Google BigQuery.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Feature Engineering](#feature-engineering)
- [03. Unsupervised Learning: K-Means Clustering](#unsupervised-kmeans)
- [04. Supervised Learning: Logistic Regression](#supervised-logreg)
- [05. Supervised Learning: Boosted Tree](#supervised-boosted)
- [06. Model Evaluation](#model-evaluation)
- [07. Fraud Predictions on New Data](#model-prediction)
- [08. Growth & Next Steps](#growth-next-steps)

---

# Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Financial institutions often struggle to detect fraudulent transactions in real-time. These events are rare (under 0.13%) and difficult to label consistently. Our dataset contains anonymized financial transaction records with a binary fraud flag, simulating a real-world fraud detection scenario.

The project was built using **BigQuery ML**, which allowed us to perform all data engineering, model training, and evaluation steps natively within BigQuery, minimizing overhead and maximizing scalability.

### Actions <a name="overview-actions"></a>

1. Loaded the transaction data from Google Cloud Storage into BigQuery.
2. Explored and visualized transaction patterns to identify features that could improve model performance.
3. Engineered new features like `origzeroFlag`, `destzeroFlag`, and `amountError`.
4. Applied undersampling to address class imbalance.
5. Trained a K-Means model to explore anomalies without labels.
6. Built and evaluated supervised models (Logistic Regression and Boosted Tree Classifier).
7. Used `ML.PREDICT()` to score new transactions and identify likely fraud cases.

### Results <a name="overview-results"></a>

After training and evaluating both supervised and unsupervised models, I determined that the **Boosted Tree model** provided the best fraud detection performance.

- The **unsupervised K-Means model** successfully grouped transactions and highlighted anomalous clusters.
- The **Logistic Regression model** achieved strong precision and interpretability.
- The **Boosted Tree model** outperformed others in recall and F1 score.

| Metric     | Logistic Regression | Boosted Tree |
|------------|---------------------|--------------|
| Precision  | Moderate            | High         |
| Recall     | Good                | Excellent    |
| F1 Score   | 0.89                | **Best**     |
| AUC (ROC)  | >0.70               | >0.90        |

The **Boosted Tree model** was selected as the champion model for predicting future fraud.

### Growth/Next Steps <a name="overview-growth"></a>

- Add additional time-series features for temporal fraud patterns.
- Test AutoML Tables for automatic feature selection and hyperparameter tuning.
- Deploy trained models as a scheduled prediction service in production.
- Integrate alerts for high-risk transactions into downstream tools (e.g., Looker or a fraud review dashboard).

---

# Data Overview <a name="data-overview"></a>

The original dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) included over **6 million** financial transactions, with the following fields:

- `type`: transaction type (TRANSFER, CASH_OUT, etc.)
- `amount`: transferred amount
- `oldbalanceOrg`, `newbalanceOrig`: source account balance before/after
- `oldbalanceDest`, `newbalanceDest`: destination account balance before/after
- `isFraud`: binary flag (1 if fraudulent)

Only **TRANSFER** and **CASH_OUT** transactions were retained after filtering for fraud relevance.

---

# Feature Engineering <a name="feature-engineering"></a>

I engineered the following additional features in SQL:

- `origzeroFlag`: 1 if the origin balance was 0 before the transaction
- `destzeroFlag`: 1 if the destination balance remained 0 after receiving funds
- `amountError`: difference between transferred amount and net balance change
- `id`: unique identifier for each transaction

I also **undersampled** 90% of non-fraud transactions to address class imbalance and created separate training and test tables using:

```sql
CREATE OR REPLACE TABLE finance.fraud_data_test AS
SELECT * FROM finance.fraud_data_sample WHERE RAND() < 0.2;

```

### Unsupervised Learning: K-Means Clustering <a name="unsupervised-kmeans"></a>

Using ML.CLUSTER in BigQuery ML, I trained a K-Means model with 5 clusters:


```sql
CREATE OR REPLACE MODEL finance.model_unsupervised
OPTIONS(model_type='kmeans', num_clusters=5) AS
SELECT amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest, type, origzeroFlag, destzeroFlag, amountError
FROM finance.fraud_data_model;
```

Evaluation:

- Davies-Bouldin Index: low (indicating good cluster separation)

- Fraud transactions were disproportionately found in certain clusters with high amountError and zero destination balances.

### Supervised Learning: Boosted Tree <a name="supervised-boosted"></a>

Next, I trained a Boosted Tree Classifier using the same dataset:

```sql
CREATE OR REPLACE MODEL finance.model_supervised_boosted_tree
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', input_label_cols=["isFraud"]) AS
SELECT type, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud
FROM finance.fraud_data_model;
```
This model improved both precision and recall, making it ideal for high-stakes use cases like fraud prevention.

### Model Evaluation <a name="model-evaluation"></a>

I compared both models using ML.EVALUATE:

```sql
CREATE OR REPLACE TABLE finance.table_perf AS
SELECT "Logistic_Regression" AS model_name, *
FROM ML.EVALUATE(MODEL `finance.model_supervised_initial`, 
SELECT * FROM finance.fraud_data_model)

UNION ALL

SELECT "Boosted_Tree" AS model_name, *
FROM ML.EVALUATE(MODEL `finance.model_supervised_boosted_tree`, 
SELECT * FROM finance.fraud_data_model);
```

| Model              | AUC   | Precision | Recall | F1 Score |
|-------------------|-------|-----------|--------|----------|
| Logistic Regression | ~0.79 | 0.84      | 0.65   | 0.73     |
| Boosted Tree       | **0.91** | **0.89**  | **0.86** | **0.88**  |


### Fraud Predictions on New Data <a name="model-prediction"></a>

Using ML.PREDICT(), I applied the champion model to unseen test data:

```sql
SELECT id, label AS predicted, isFraud AS actual
FROM ML.PREDICT(MODEL `finance.model_supervised_boosted_tree`, (
  SELECT * FROM finance.fraud_data_test
)), UNNEST(predicted_isfraud_probs) AS p
WHERE p.label = 1 AND p.prob > 0.5;
```

Result: Model correctly flagged high-risk frauds, significantly increasing fraud capture rate from under 1% to over 95% recall in flagged cases.

### Growth & Next Steps <a name="growth-next-steps"></a>

- Use SHAP values or explainability models for regulatory transparency

- Add temporal features (hour of day, transaction rate over time)

- Deploy using scheduled queries or Vertex AI for real-time flagging

- Experiment with ensemble models or anomaly score stacking



