# Customer Conversion Prediction using Machine Learning 🎯📈

## Overview

This project focuses on predicting customer conversion using machine learning classification models.

The objective is to determine whether a customer is likely to convert based on demographic information, marketing campaign interactions, website engagement metrics, and historical purchasing behavior.

Multiple classification algorithms were implemented and compared to identify the most effective approach for predicting customer conversions.

---

## Business Problem

Marketing campaigns often target thousands of customers.

However, only a small percentage of customers ultimately convert.

Accurately predicting conversion likelihood can help businesses:

* Improve marketing efficiency
* Reduce advertising costs
* Increase conversion rates
* Prioritize high-value leads
* Optimize customer targeting strategies

This project aims to build predictive models that estimate the probability of customer conversion based on behavioral and campaign-related data.

---

## Dataset Features

The dataset contains information related to customer demographics, marketing campaigns, and customer engagement.

### Demographic Information

* Age
* Gender
* Income

### Marketing Campaign Information

* Campaign Channel
* Campaign Type
* Ad Spend
* Click Through Rate (CTR)

### Customer Engagement Metrics

* Website Visits
* Pages Per Visit
* Time On Site
* Social Shares
* Email Opens
* Email Clicks

### Historical Customer Data

* Previous Purchases
* Loyalty Points

### Target Variable

```text
Conversion
```

* 1 = Customer Converted
* 0 = Customer Did Not Convert

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

---

## Data Preprocessing

The following preprocessing steps were performed:

### Missing Value Inspection

The dataset was checked for null values and cleaned where necessary.

---

### Label Encoding

Gender was converted into numerical format:

```python
encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])
```

---

### One-Hot Encoding

Categorical marketing variables were encoded:

```python
pd.get_dummies(
    df,
    columns=[
        "CampaignChannel",
        "CampaignType"
    ]
)
```

This allowed machine learning algorithms to process categorical campaign information effectively.

---

### Feature Selection

The following columns were removed:

```text
CustomerID
AdvertisingPlatform
AdvertisingTool
```

These variables either served as identifiers or contained limited predictive value.

---

### Feature Scaling

Since Logistic Regression is sensitive to feature magnitudes, StandardScaler was applied.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Exploratory Data Analysis

Several visualizations were created to better understand customer behavior and conversion patterns.

### Analysis Performed

* Conversion Distribution
* Correlation Heatmaps
* Feature Importance Exploration
* ROC Curve Visualization
* Confusion Matrix Analysis
* Customer Engagement Analysis

---

## Models Implemented

### 1. Logistic Regression

Used as the primary baseline classification model.

Logistic Regression estimates the probability of customer conversion and provides interpretable results.

---

### 2. Balanced Logistic Regression

Class weights were adjusted to handle class imbalance:

```python
class_weight="balanced"
```

This approach aimed to improve performance on minority classes.

---

### 3. Decision Tree Classifier

A Decision Tree model was trained and evaluated to capture nonlinear decision boundaries and compare performance against Logistic Regression.

---

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## Results

### Logistic Regression

```text
Accuracy ≈ 89%
ROC-AUC ≈ 0.79
```

---

### Decision Tree Classifier

```text
Accuracy ≈ 87%
ROC-AUC ≈ 0.74
```

---

## Best Model

### Logistic Regression

Logistic Regression achieved the strongest overall performance.

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | ~89%     | ~0.79   |
| Decision Tree       | ~87%     | ~0.74   |

Although Decision Trees can capture nonlinear patterns, Logistic Regression generalized better on unseen data and achieved a higher ROC-AUC score.

---

## Feature Engineering Experiments

Several feature engineering experiments were conducted to improve model performance.

Examples included:

* Scaling numerical features
* Removing highly predictive variables
* Evaluating class balancing strategies
* Comparing model robustness across feature subsets

One notable experiment involved removing the Conversion Rate feature to evaluate potential target leakage and model dependency.

Despite a slight performance decrease, the model maintained strong predictive capability.

---

## Key Insights

### Customer Engagement Drives Conversion

Features such as:

* Email Opens
* Email Clicks
* Website Visits
* Time On Site

were strongly associated with conversion likelihood.

---

### Historical Purchasing Behavior Matters

Customers with:

* Previous Purchases
* Loyalty Points

demonstrated higher conversion probabilities.

---

### Marketing Campaigns Influence Outcomes

Campaign type and communication channel significantly impacted customer conversion rates.

---

### Logistic Regression Remains Competitive

Despite being a relatively simple model, Logistic Regression outperformed the more complex Decision Tree model on this dataset.

This suggests that customer conversion behavior can often be modeled effectively using smooth probabilistic relationships.

---

## Project Structure

```text
customer-conversion-prediction/
│
├── notebooks/
│   ├── Logistic_Regression.ipynb
│   └── Decision_Tree.ipynb
│
├── images/
│
├── README.md
│
└── requirements.txt
```

---

## Future Improvements

Potential future enhancements include:

* Random Forest Classifier
* XGBoost Classifier
* Support Vector Machines (SVM)
* Ensemble Learning Techniques
* Hyperparameter Optimization
* Flask Deployment
* Real-Time Lead Scoring Dashboard

---

## Skills Demonstrated

This project demonstrates practical experience with:

* Data Cleaning
* Feature Engineering
* Classification Modeling
* Logistic Regression
* Decision Trees
* ROC-AUC Analysis
* Model Comparison
* Feature Scaling
* Customer Analytics
* Marketing Analytics

---

## Conclusion

This project explored multiple machine learning approaches for predicting customer conversions.

Logistic Regression achieved the strongest performance, demonstrating that customer engagement and marketing interaction data contain meaningful signals that can be used to predict conversion behavior.

The project highlights the importance of feature engineering, model evaluation, and algorithm comparison when building business-focused predictive systems.

---

## Author

Built as part of a machine learning learning journey focused on:

* Classification Modeling
* Marketing Analytics
* Customer Behavior Prediction
* Feature Engineering
* Applied Machine Learning

