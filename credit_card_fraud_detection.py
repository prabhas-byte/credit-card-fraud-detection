# ==================== CREDIT CARD FRAUD DETECTION ====================
# Task 3 - Complete Script (Easy to understand)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. LOAD YOUR DATASET ======================
# Change the filename if your file has a different name
df = pd.read_csv('creditcard.csv')        # ←←← CHANGE THIS IF NEEDED

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df['Class'].value_counts())
print(f" Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

# ====================== 2. PREPROCESSING ======================
# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the data (very important for this dataset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData split done!")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ====================== 3. TRAIN THE MODEL ======================
# Using Random Forest with balanced class weights (handles imbalance)
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',   # This helps with fraud imbalance
    random_state=42,
    n_jobs=-1
)

print("\nTraining model... (this may take 10-30 seconds)")
model.fit(X_train, y_train)

# ====================== 4. MAKE PREDICTIONS ======================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ====================== 5. EVALUATE THE MODEL ======================
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC-AUC Score (very important for fraud detection)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {auc:.4f} (Higher is better)")

# ====================== 6. VISUALIZATIONS ======================

# Plot 1: Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Genuine', 'Fraud'], 
            yticklabels=['Genuine', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot 2: Feature Importance (Top 10)
importances = pd.Series(model.feature_importances_, index=df.drop('Class', axis=1).columns)
importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Top 10 Most Important Features for Fraud Detection')
plt.xlabel('Importance Score')
plt.show()

print("\n" + "="*60)
print("Task Completed Successfully!")
print("You can now submit this script + graphs.")
print("="*60)