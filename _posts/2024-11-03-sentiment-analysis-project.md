
---
layout: post
title: How Machine Learning Transforms Sentiment Analysis
image: "/posts/thumbnail.png"
tags: [Sentiment Analysis, NLP, Machine Learning, Python, Text Classification, Logistic Regression, Transformers]
---

This project explores how machine learning can be applied to text data for **sentiment analysis**. Using the IMDb movie review dataset, I built and evaluated multiple classification models to predict whether a review expresses a positive or negative sentiment.

# Table of contents

- [00. Introduction](#intro)
- [01. Dataset Overview](#dataset)
- [02. Preprocessing](#preprocessing)
- [03. Feature Extraction](#features)
- [04. Model Building](#model)
- [05. Evaluation](#evaluation)
- [06. Comparing Other Models](#comparison)
- [07. Challenges & Observations](#challenges)
- [08. Next Steps & Improvements](#next-steps)
- [09. Conclusion](#conclusion)

___

# Introduction <a name="intro"></a>

Deciphering large amounts of textual data can be daunting. I often found it difficult to extract meaningful insights from open-ended responses during market research projects. This led me to explore **machine learning for sentiment analysis** — a powerful tool to automate and scale the understanding of customer emotions.

![Pipeline](/posts/model.png)

---

# Dataset Overview <a name="dataset"></a>

The IMDb dataset is a benchmark in sentiment analysis, with **50,000 movie reviews**, split evenly between **positive** and **negative** sentiments.

**Key Stats:**
- Number of reviews: 50,000  
- Average length: ~144 words  
- Common words: *love*, *boring*, *heartwarming*

![Sentiment Distribution](/posts/distribution-sentiment-labels.png)
![Word Cloud](/posts/word-cloud.png)

---

# Preprocessing <a name="preprocessing"></a>

To improve model performance, I applied:

- **Tokenization**
- **Stopword Removal**
- **Lemmatization**

```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def preprocess(text):
    tokens = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])
```

![Preprocessing Code](/posts/data-processing.png)

---

# Feature Extraction <a name="features"></a>

I used **TF-IDF** (Term Frequency–Inverse Document Frequency) to convert text into numerical vectors, highlighting important words based on frequency.

---

# Model Building <a name="model"></a>

My baseline model was **Logistic Regression**, chosen for its simplicity and effectiveness.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_val, y_val)
```

![Model Code](/posts/model.png)

---

# Evaluation <a name="evaluation"></a>

### Cross-Validation Results:

- Validation Accuracy: **87.28%**
- Test Accuracy: **84.88%**
- ROC AUC Score: **0.94**

![Accuracy Distribution](/posts/distribution.png)
![Cross-Validation Bar Chart](/posts/cross-val.png)
![ROC Curve](/posts/roc-curve.png)

---

# Comparing Other Models <a name="comparison"></a>

I compared logistic regression with:

- **MLP** (Accuracy: 85.59%)
- **Naive Bayes** (Accuracy: 86.64%)

These findings suggest that even simple models can perform well when preprocessing is thorough.

---

# Challenges & Observations <a name="challenges"></a>

- **Handling Long Reviews**: Preserving context was tricky  
- **Class Imbalance**: Required maintaining equal label distribution  
- **Mixed Sentiments**: Ambiguous reviews made modeling harder

---

# Next Steps & Improvements <a name="next-steps"></a>

For more nuanced analysis (e.g., sarcasm, emotion), future work could explore:

- **Transformer models** (BERT, GPT)
- Real-time feedback engines
- Enhanced deployment pipelines for chatbots and social monitoring

---

# Conclusion <a name="conclusion"></a>

Machine learning has reshaped how we approach sentiment analysis. This project proved that even simple models like Logistic Regression can produce strong results when supported by proper preprocessing and feature engineering.

*For additional resources, see Google’s [text classification guide](https://developers.google.com/machine-learning/guides/text-classification).*
