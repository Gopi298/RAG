import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. Load Dataset
df = pd.read_csv('diabetes.csv')

# 2. Data Cleaning: Replace zero values in physiological features with medians
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 3. Feature Engineering
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]).astype(float)
df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
df['Age_BMI_Product'] = df['Age'] * df['BMI']

# 4. Separate Features & Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 5. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Model Training (Optimized Regularized Random Forest)
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 8. Evaluation
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)
test_proba = model.predict_proba(X_test_scaled)[:, 1]

print("--- Training Metrics ---")
print(f"Train Accuracy: {accuracy_score(y_train, train_preds) * 100:.2f}%")
print(f"Test Accuracy:  {accuracy_score(y_test, test_preds) * 100:.2f}%")
print(f"ROC-AUC Score:  {roc_auc_score(y_test, test_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_preds))

# 9. Save Artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved model.pkl and scaler.pkl successfully!")
