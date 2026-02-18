"""
Example: Training a Machine Learning Model on Preprocessed Network Flow Data
This script demonstrates how to use the preprocessed data for intrusion detection.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv('test_preprocessed.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Separate features and labels
label_columns = ['Label', 'Attack_Category', 'Attack_Subcategory']
X = df.drop(columns=label_columns)
y = df['Label']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Label distribution:\n{y.value_counts()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train Random Forest classifier
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"Model Performance")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print(f"\n{'='*60}")
print(f"Top 10 Most Important Features")
print(f"{'='*60}")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

print(f"\n{'='*60}")
print(f"Model training complete!")
print(f"{'='*60}")
