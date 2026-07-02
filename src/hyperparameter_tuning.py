import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("data/Titanic-Dataset.csv")

# Data Cleaning
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Handle Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode Categorical Variables
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# BASELINE MODEL
# =========================
baseline_rf = RandomForestClassifier(random_state=42)

baseline_rf.fit(X_train, y_train)

baseline_pred = baseline_rf.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("===================================")
print("BASELINE MODEL RESULTS")
print("===================================")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# =========================
# HYPERPARAMETER TUNING
# =========================

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

tuned_accuracy = accuracy_score(y_test, y_pred)

# =========================
# RESULTS
# =========================

print("\n===================================")
print("GRID SEARCH RESULTS")
print("===================================")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(round(grid_search.best_score_, 4))

print("\nTuned Model Accuracy:")
print(round(tuned_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n===================================")
print("COMPARISON")
print("===================================")

print(f"Baseline Accuracy : {baseline_accuracy:.4f}")
print(f"Tuned Accuracy    : {tuned_accuracy:.4f}")

# =========================
# SAVE RESULTS
# =========================

with open("results.txt", "w") as f:
    f.write("Hyperparameter Tuning Results\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Baseline Accuracy: {baseline_accuracy:.4f}\n\n")

    f.write("Best Parameters:\n")
    f.write(str(grid_search.best_params_))
    f.write("\n\n")

    f.write(f"Best CV Score: {grid_search.best_score_:.4f}\n")
    f.write(f"Tuned Accuracy: {tuned_accuracy:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\nResults saved to results.txt")