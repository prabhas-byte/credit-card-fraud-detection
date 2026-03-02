# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using a highly imbalanced real-world dataset.

Built with Python, pandas, scikit-learn (Random Forest), and visualization tools (matplotlib & seaborn).

### Project Goal
Build a classification model that identifies fraudulent transactions (Class = 1) while minimizing false positives and maximizing fraud detection (high recall).

### Dataset
- Source: Kaggle Credit Card Fraud Detection dataset
- ~284,807 transactions
- Extremely imbalanced (~0.17% fraud cases)
- Features: V1–V28 (PCA-transformed), Time, Amount, Class (0 = genuine, 1 = fraud)

### What the project does
- Loads and explores the dataset
- Scales features (StandardScaler)
- Handles class imbalance using class_weight='balanced'
- Trains a Random Forest Classifier
- Evaluates using precision, recall, F1-score, ROC-AUC, confusion matrix
- Shows top important features

### Key Results (typical)
- ROC-AUC: 0.95+
- Fraud recall: ~0.80–0.90
- Precision for fraud: ~0.70–0.85
- Very low false positives on genuine transactions

### How to Run

1. Clone or download this repository
2. Place `creditcard.csv` in the same folder (download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
3. Install required packages

## Run the Script

python filename.py

### license

MIT License – free to use, modify, and learn from the task.

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
