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

### Growth/Next Steps <a name="overview-growth"></a>

Although model performance was excellent, especially with XGBoost, further improvements and extensions could include:

- Applying advanced **hyperparameter tuning** (GridSearch, Optuna, or Bayesian optimization)
- Testing **oversampling techniques** like SMOTE/ADASYN to compare with class weights
- Exploring **deep learning models** or **AutoML** platforms for further automation
- Deploying the model into a **real-time fraud detection pipeline** with drift monitoring
- Experimenting with time-based features (e.g., rolling transaction behavior, user patterns)

<br>
<br>

### Key Definition  <a name="overview-definition"></a>

The *Class* variable in our dataset is the target outcome:

- A value of **0** represents a **legitimate** transaction  
- A value of **1** represents a **fraudulent** transaction
  
<be>

My goal was to accurately identify transactions labeled as `1` (fraud), without misclassifying too many legitimate ones. A well-performing model in this context is one that maximizes **recall** and **precision** for the minority class, while minimizing the cost of false positives.
