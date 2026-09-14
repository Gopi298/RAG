import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# 1. Load Data
df = pd.read_csv('diabetes.csv')

# 2. Handle Zero Values (Replace invalid 0s with column medians)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 3. Feature Engineering
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]).astype(float)
df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
df['Age_BMI_Product'] = df['Age'] * df['BMI']

# 4. Features & Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training (Optimized Random Forest for maximum stability)
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluation
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)
test_proba = model.predict_proba(X_test_scaled)[:, 1]

print(f"Train Accuracy: {accuracy_score(y_train, train_preds)*100:.2f}%")
print(f"Test Accuracy:  {accuracy_score(y_test, test_preds)*100:.2f}%")
print(f"ROC AUC Score:  {roc_auc_score(y_test, test_proba):.4f}")

# Save artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler successfully saved!")
