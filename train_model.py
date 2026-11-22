# DCS Project - Train ML Model (Random Forest)
# Run this after collecting real data in data/signal_log.csv

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

os.makedirs("ml", exist_ok=True)

# Load data
CSV_PATH = "data/signal_log.csv"
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found. Run simulate_jamming.py first or collect real data.")
    exit()

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# Features and target
X = df[["rssi", "snr", "loss_rate"]].values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Jammed"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
features = ["rssi", "snr", "loss_rate"]
importances = model.feature_importances_
print("\nFeature Importances:")
for f, i in zip(features, importances):
    print(f"  {f}: {i:.4f}")

# Save model
joblib.dump(model, "ml/model.pkl")
print("\nModel saved to ml/model.pkl")
