# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).  

This project demonstrates data exploration, handling class imbalance with SMOTE, baseline modeling, and evaluation using classification metrics and visualizations.

---

## 🧩 Project Structure

fraud-detection/
│── data/ # Dataset (not included in repo)
│── notebooks/ # Jupyter notebooks
│ └── fraud_detection_mvp.ipynb
│── src/ Nothing in here yet
│── README.md
│── .gitignore


---

## 🛠️ Technologies & Libraries

- **Python 3.10**
- **Jupyter Notebook**
- **pandas, numpy** → data handling
- **scikit-learn** → machine learning
- **imbalanced-learn** → SMOTE for class imbalance
- **matplotlib, seaborn** → data visualization

---

## 📊 Phases

1. **Data Loading & Inspection**  
   - Loaded the dataset and checked structure and first rows.

2. **Data Exploration**  
   - Counted fraud vs non-fraud transactions  
   - Calculated percentage of fraud cases

3. **Train/Test Split**  
   - Stratified split to maintain class distribution

4. **Baseline Model: Logistic Regression**  
   - Trained on raw data  
   - Evaluated using precision, recall, F1-score, ROC-AUC

5. **Handling Class Imbalance (SMOTE)**  
   - Scaled features with `StandardScaler`  
   - Oversampled minority class (fraud) using SMOTE  
   - Retrained Logistic Regression  
   - Visualized confusion matrix and ROC curve  

---

⚠️ Notes

The dataset is highly imbalanced: ~0.17% fraud cases.

Accuracy alone is misleading — ROC-AUC, precision, and recall are more informative.

CSV file is not included in the repo due to size and sensitivity

🔗 Dataset

https://www.kaggle.com/mlg-ulb/creditcardfraud

