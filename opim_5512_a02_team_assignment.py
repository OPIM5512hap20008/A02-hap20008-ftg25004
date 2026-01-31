# -*- coding: utf-8 -*-
"""OPIM 5512 A02 Team Assignment
MLPRegressor with improved preprocessing, hyperparameters, and visualization
"""

from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load California Housing dataset
# -----------------------------
housing = fetch_california_housing(as_frame=True)

# Combine features and target into one DataFrame for easier handling
df = housing.frame

# Separate predictors and response variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# -----------------------------
# Train-test split
# -----------------------------
# Using a fixed random_state ensures reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Feature scaling (KEY IMPROVEMENT)
# -----------------------------
# Neural networks are sensitive to feature scale.
# Standardizing features improves convergence speed and model stability.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data to avoid data leakage
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Improved MLPRegressor model
# -----------------------------
mlp = MLPRegressor(
    # Smaller, more balanced hidden layers help reduce overfitting
    hidden_layer_sizes=(64, 32),
    
    # ReLU is efficient and works well for non-linear regression problems
    activation='relu',
    
    # Adam optimizer provides faster and more stable convergence
    solver='adam',
    
    # Explicit learning rate improves training control
    learning_rate_init=0.001,
    
    # Increased max_iter allows the model enough time to converge
    max_iter=2000,
    
    # Early stopping prevents overfitting by monitoring validation loss
    early_stopping=True,
    validation_fraction=0.1,
    
    # Tighter tolerance improves convergence accuracy
    tol=1e-4,
    
    # Fixed random_state ensures consistent results
    random_state=42
)

# Train the model on scaled training data
mlp.fit(X_train_scaled, y_train)

# -----------------------------
# Predictions
# -----------------------------
# Generate predictions for both training and test sets
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)

# -----------------------------
# Model evaluation (IMPROVEMENT)
# -----------------------------
# RMSE provides an interpretable error metric in original units
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# R² measures goodness-of-fit and allows comparison across models
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {train_rmse:.3f}")
print(f"Test RMSE:     {test_rmse:.3f}")
print(f"Training R²:   {train_r2:.3f}")
print(f"Test R²:       {test_r2:.3f}")

# -----------------------------
# Visualization: Actual vs Predicted (Training Data)
# -----------------------------
plt.figure(figsize=(8, 6))

# Scatter plot shows how closely predictions match actual values
plt.scatter(y_train, y_train_pred, alpha=0.5)

# 45-degree reference line represents perfect predictions
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    linestyle='--',
    linewidth=2
)

# Clear labels and larger fonts improve interpretability
plt.xlabel('Actual Median House Value', fontsize=12)
plt.ylabel('Predicted Median House Value', fontsize=12)
plt.title('MLPRegressor: Actual vs Predicted (Training Data)', fontsize=14)

# Grid improves readability and comparison of values
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization: Actual vs Predicted (Test Data)
# -----------------------------
plt.figure(figsize=(8, 6))

# Separate test plot ensures model generalization is evaluated independently
plt.scatter(y_test, y_test_pred, alpha=0.5)

# Same reference line allows direct comparison with training plot
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--',
    linewidth=2
)

plt.xlabel('Actual Median House Value', fontsize=12)
plt.ylabel('Predicted Median House Value', fontsize=12)
plt.title('MLPRegressor: Actual vs Predicted (Test Data)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()