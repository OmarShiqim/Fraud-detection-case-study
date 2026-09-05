# Credit Card Fraud Detection — Case Study

A machine learning case study classifying credit card transactions as fraudulent or 
legitimate, using the BankSim-style transaction dataset (594,643 transactions, 1.21% fraud rate).

## Overview

This project walks through the full pipeline: data cleaning, exploratory data analysis, 
feature engineering, handling severe class imbalance, model comparison, hyperparameter 
tuning, and business recommendations.

## Dataset

- **Source:** BankSim synthetic transaction dataset (`fraud.csv`)
- **Size:** 594,643 transactions
- **Features:** step, customer, age, gender, merchant, category, amount, fraud (target)
- **Class imbalance:** only 1.21% of transactions are fraudulent

## Approach

1. **Data cleaning** — stripped quote-wrapped fields, dropped constant zip columns
2. **EDA** — identified category and transaction amount as the strongest fraud signals; 
   confirmed age/gender are statistically negligible predictors (Cramér's V < 0.03); 
   confirmed outliers in amount are signal, not noise (26% fraud rate among outliers vs. 
   1.2% baseline)
3. **Feature engineering** — one-hot encoding, customer-level spend aggregates (built 
   without referencing the fraud label to avoid data leakage)
4. **Train/test split** — stratified 80/20 split performed *before* any resampling
5. **Class imbalance handling** — SMOTE applied to the training set only
6. **Modeling** — compared Logistic Regression, Random Forest, and Gradient Boosting
7. **Evaluation metric** — AUC-PR (rather than accuracy or AUC-ROC), since the latter two 
   are uninformative at this level of class imbalance
8. **Hyperparameter tuning** — RandomizedSearchCV on a subsample, refit on full training data

## Results

| Model | Recall | Precision | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| Logistic Regression | 95% | 27% | ~0.994 | 0.821 |
| Random Forest | 86% | 63% | ~0.994 | 0.826 |
| **Gradient Boosting (selected)** | **96%** | **30%** | **0.996** | **0.871** |

**Final model:** Tuned Gradient Boosting — catches 96% of fraud cases at 30% precision.

## Business Recommendation

Deploy with a human-in-the-loop review process rather than automatic transaction blocking, 
given the current false-alarm rate. Prioritize the review queue by transaction amount and 
high-risk category (leisure, travel). Revisit the model's decision threshold once real 
costs of false alarms vs. missed fraud are measured from live data.

## Repository Contents

- `fraud_detection_case_study.ipynb` — full analysis and modeling notebook
  
## How to Run

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection_case_study.ipynb
```

**Dependencies:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib
