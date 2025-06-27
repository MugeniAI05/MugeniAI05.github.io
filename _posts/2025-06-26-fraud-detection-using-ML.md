---
layout: post
title: "Fraud Detection Using Machine Learning: Logistic Regression, Random Forest, and XGBoost"
image: "/posts/fraud-detection.jpeg"
tags: [Fraud Detection, Machine Learning, balanced Data, Python, Credit Card]
---

A financial institution is trying to reduce credit card fraud by detecting fraudulent transactions from their large transaction datasets. Fraudulent activity is rare, making up less than 0.2% of all records. This project demonstrates how multiple machine learning models were applied to classify fraud accurately.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Logistic Regression](#logreg-title)
- [04. XGBoost Classifier](#xgboost-title)
- [05. Random Forest](#rf-title)
- [06. Modelling Summary](#modelling-summary)
- [07. Visualizations](#visualizations)
- [08. Growth & Next Steps](#growth-next-steps)
___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

We used a publicly available dataset of credit card transactions to build a machine learning pipeline for fraud detection. Fraudulent transactions are rare but costly — identifying them quickly and accurately can save institutions millions.

The key challenge in this project was working with a highly **imbalanced dataset** where only a small fraction of observations were fraud. This meant that traditional accuracy metrics weren’t sufficient, and we had to focus on F1-score, precision, recall, and AUC.

We aimed to build models that can generalize well and flag suspicious transactions, helping stakeholders take timely action to prevent financial losses.

<br>
<br>

### Actions <a name="overview-actions"></a>

We began by performing **data preprocessing**, including:

- Handling missing values
- Dropping ID columns
- Scaling numerical features (e.g., `Amount`) using `StandardScaler`
- Splitting the dataset into training and test sets

We also addressed the **class imbalance** using methods like:

- `class_weight='balanced'` in Logistic Regression
- `scale_pos_weight` in XGBoost

We trained and evaluated three classification models:

* Logistic Regression  
* XGBoost Classifier  
* Random Forest  

Each model was evaluated using 5-fold cross-validation with **F1-score** as the main metric and validated using **ROC curves** and **confusion matrices** on the test set.

<br>
<br>

### Results <a name="overview-results"></a>

Our testing found that the XGBoost Classifier had the highest predictive accuracy.

<br>
**Metric 1: F1 Score (Cross-Validation)**

* Logistic Regression = 0.964  
* Random Forest = 0.985  
* XGBoost = **0.999**

<br>
**Metric 2: ROC AUC (Test Set)**

* Logistic Regression = 0.97  
* Random Forest = 0.99  
* XGBoost = **1.00**

Given that fraud detection prioritizes minimizing false negatives while balancing false positives, we used **F1-score** and **AUC** as the primary evaluation metrics. XGBoost achieved near-perfect performance on both metrics, making it the model of choice for deployment.

<br>
<br>

### Key Definition  <a name="overview-definition"></a>

The *Class* variable in our dataset is the target outcome:

- A value of **0** represents a **legitimate** transaction  
- A value of **1** represents a **fraudulent** transaction
  
Our goal was to accurately identify transactions labeled as `1` (fraud), without misclassifying too many legitimate ones. A well-performing model in this context is one that maximizes **recall** and **precision** for the minority class, while minimizing the cost of false positives.

___

# Data Overview  <a name="data-overview"></a>

The dataset contains **568,630** credit card transactions across **30 anonymized features** (`V1` through `V28`, `Amount`, and `Time`). The target variable is `Class`, where:

- `0` = Legitimate transaction  
- `1` = Fraudulent transaction

Key points:

- No missing values were found in the dataset  
- The `Amount` variable ranges from $50 to over $24,000  
- Features were scaled using `StandardScaler`
- The dataset was split into 80% training and 20% testing

```python
df = pd.read_csv("creditcard_2023.csv")
df.info()
df.Amount.describe()
df.isnull().sum()

# Drop ID and target column
X = df.drop(["id", "Class"], axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

___

# Modelling Overview <a name="modelling-overview"></a>

We built and evaluated three models:

- Logistic Regression  
- XGBoost Classifier  
- Random Forest  

Evaluation was based on:

- F1 Score (cross-validation)  
- Classification report (precision, recall)  
- ROC AUC Score and ROC curve  
- Feature importance (for Random Forest)

___

# Logistic Regression <a name="logreg-title"></a>

Logistic Regression was used as a baseline model. We applied class weighting to handle imbalance.

```python
log_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

cv_scores_log = cross_val_score(log_model, X_train_scaled, y_train, cv=5, scoring='f1')
log_model.fit(X_train_scaled, y_train)
log_y_pred = log_model.predict(X_test_scaled)
print(classification_report(y_test, log_y_pred))
```

**Cross-validated F1 Score**: 0.964  
**ROC AUC**: 0.97

___

# XGBoost Classifier <a name="xgboost-title"></a>

XGBoost handled class imbalance with `scale_pos_weight`.

```python
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42
)

cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='f1')
xgb_model.fit(X_train_scaled, y_train)
xgb_y_pred = xgb_model.predict(X_test_scaled)
print(classification_report(y_test, xgb_y_pred))
```

**Cross-validated F1 Score**: 0.999  
**ROC AUC**: 1.00

___

# Random Forest <a name="rf-title"></a>

Random Forest was tuned with `max_depth=10` and `min_samples_split=5`.

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))
```

**Cross-validated F1 Score**: 0.985  
**ROC AUC**: 0.99

___

# Modelling Summary  <a name="modelling-summary"></a>

The goal of this project was to detect fraudulent transactions accurately, with special focus on **precision**, **recall**, and **AUC** due to the class imbalance.

<br>
**Metric 1: F1 Score (Cross-Validation)**

* Logistic Regression = 0.964  
* Random Forest = 0.985  
* XGBoost = **0.999**

<br>
**Metric 2: ROC AUC (Test Set)**

* Logistic Regression = 0.97  
* Random Forest = 0.99  
* XGBoost = **1.00**

XGBoost emerged as the best-performing model. While all models performed well, XGBoost achieved near-perfect precision and recall on the test set, making it ideal for deployment.

___

# Visualizations <a name="visualizations"></a>

### ROC Curves

Visual inspection of ROC curves confirms XGBoost’s superiority, closely followed by Random Forest.

![ROC Curve - Logistic Regression](/img/posts/roc-curve-lg.png)
![ROC Curve - XGBoost](/img/posts/roc-curve-xgb.png)
![ROC Curve - Random Forest](/img/posts/roc-curve-rf.png)

<br>

### Confusion Matrix

Random Forest and XGBoost had minimal misclassifications.

![Confusion Matrix - Random Forest](/img/posts/confusion-matrix-rf.png)

<br>

### Feature Importance

Random Forest provided insight into which features were most influential in detecting fraud.

![Feature Importance - Random Forest](/img/posts/feature-importance-ranking-rf.png)

<br>

### Correlation Heatmap

We also visualized correlation between features to check for multicollinearity.

![Correlation Heatmap](/img/posts/feature-corr-heatmap-rf.png)

___

### Growth/Next Steps <a name="overview-growth"></a>

Although model performance was excellent, especially with XGBoost, further improvements and extensions could include:

- Applying advanced **hyperparameter tuning** (GridSearch, Optuna, or Bayesian optimization)
- Testing **oversampling techniques** like SMOTE/ADASYN to compare with class weights
- Exploring **deep learning models** or **AutoML** platforms for further automation
- Deploying the model into a **real-time fraud detection pipeline** with drift monitoring
- Experimenting with time-based features (e.g., rolling transaction behavior, user patterns)

