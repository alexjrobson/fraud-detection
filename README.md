# Credit Card Fraud Detection

This repository contains a machine learning project to detect fraudulent credit card transactions using the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. The project demonstrates a full workflow from data exploration and preprocessing to model training, evaluation, and comparison.

---

## Project Overview

Credit card fraud is a critical problem in the financial industry. Detecting fraud is challenging due to the **extreme imbalance** in the dataset: fraudulent transactions are very rare (~0.17% of all transactions). This project explores techniques to handle imbalanced datasets and evaluates model performance on key metrics.

---

## Dataset

- File: `creditcard.csv`
- Features: 30 numeric features (`V1`–`V28`, `Time`, `Amount`)
- Target: `Class` (0 = non-fraud, 1 = fraud)
- Dataset size: 284,807 transactions

---

## Workflow

### Phase 4: Logistic Regression with SMOTE

- Train Logistic Regression on the dataset.
- Scale features using `StandardScaler`.
- Handle class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).
- Evaluate using:
  - Precision, Recall, F1-score
  - ROC-AUC
- Visualize:
  - Confusion matrix
  - ROC curve

**Insight:** Scaling and SMOTE improve the model's ability to detect fraud compared to training on the imbalanced dataset.

---

### Phase 5: Random Forest + SMOTE

- Train a Random Forest classifier on the SMOTE-resampled training set.
- Hyperparameters:
  - 200 estimators
  - Maximum depth of 10
- Evaluate using the same metrics as Logistic Regression.
- Visualize:
  - Confusion matrix
  - ROC curve

**Insight:** Random Forest increases **recall** for fraud detection, identifying more fraudulent transactions, though with slightly lower precision.

---

### Phase 6: Model Comparison & Recall Visualization

- Compare all models for the fraud class:
  - Logistic Regression (Baseline)
  - Logistic Regression + SMOTE
  - Random Forest + SMOTE
- Metrics considered: Precision, Recall, F1-score, ROC-AUC
- Visualize **Recall** using a bar plot to highlight which model detects the most fraud.

**Insight:** Demonstrates the trade-off between precision and recall and highlights the effectiveness of SMOTE and Random Forest in detecting fraud.

**Comparison Table (Fraud Class)**

| Model | Precision | Recall | F1-score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression (Baseline) | 0.83 | 0.69 | 0.76 | 0.9487 |
| Logistic Regression + SMOTE | 0.82 | 0.66 | 0.73 | 0.9512 |
| Random Forest + SMOTE | 0.43 | 0.86 | 0.57 | 0.9820 |

---

## Visualizations

- **Confusion Matrix:** Shows True Positives, True Negatives, False Positives, and False Negatives for each model.
- **ROC Curve:** Plots True Positive Rate vs False Positive Rate across thresholds.
- **Recall Bar Plot:** Highlights which model detects the most fraud.

---

## Key Takeaways

- Fraud detection is challenging due to extreme class imbalance.
- Logistic Regression with SMOTE improves detection over baseline Logistic Regression.
- Random Forest with SMOTE further increases recall, catching more fraudulent transactions.
- Precision decreases with higher recall due to more false positives — a classic trade-off in fraud detection.
- ROC-AUC scores indicate strong model separation capability for both Logistic Regression + SMOTE and Random Forest + SMOTE.

---

## Requirements

- Python 3.0
- Packages:
  - pandas
  - scikit-learn
  - imbalanced-learn
  - matplotlib
  - seaborn
  - jupyter notebook or JupyterLab

You can install dependencies using:

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn
