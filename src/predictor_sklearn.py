"""
Algorithm Execution Time Predictor - Scikit-Learn Implementation
This script uses Scikit-Learn to implement Polynomial Regression.
It solves the "Underfitting" problem seen in the linear model by automatically 
generating polynomial and interaction features to capture O(n^2) time complexity.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. LOAD DATASET
# ==========================================
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '../data/algo_data.csv')
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)

# X features: array_size, unsorted_pct, nested_loops, cpu_load
x_train = data[:, :4]
# Y target: execution_time (ms)
y_train = data[:, 4]

# ==========================================
# 2. AUTOMATED FEATURE ENGINEERING
# ==========================================
# Instead of manually squaring features, PolynomialFeatures automatically creates 
# all combinations (e.g., array_size^2, array_size * nested_loops). 
# This helps the model understand complex, non-linear algorithm behaviors.
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x_train)

# ==========================================
# 3. FEATURE SCALING
# ==========================================
# Standardizes features by removing the mean and scaling to unit variance (Z-score).
# Essential for numerical stability, especially after generating large polynomial features.
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_poly)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
# Scikit-Learn's LinearRegression uses highly optimized solvers (like SVD) 
# instead of manual Gradient Descent, making it incredibly fast and accurate.
model = LinearRegression()
model.fit(x_norm, y_train)

# ==========================================
# 5. EVALUATION
# ==========================================
predictions = model.predict(x_norm)
mse = mean_squared_error(y_train, predictions)
print(f"Cost (MSE) using Scikit-Learn: {mse:.2f}")

# ==========================================
# 6. MAKE A PREDICTION ON UNSEEN DATA
# ==========================================
# New instance: [array_size, unsorted_pct, nested_loops, cpu_load]
x_new = np.array([[15000, 70, 2, 6]])

# Crucial Step: The new data must pass through the EXACT same pipeline
x_new_poly = poly.transform(x_new)       # 1. Generate polynomial features
x_new_norm = scaler.transform(x_new_poly)  # 2. Scale using the fitted scaler

predicted_time = model.predict(x_new_norm)
print(f"Predicted Execution Time: {predicted_time[0]:.2f} ms")

# ==========================================
# 7. DATA VISUALIZATION
# ==========================================
plt.figure(figsize=(8, 6))

# Plot actual vs predicted values
plt.scatter(y_train, predictions, color='green', alpha=0.5, label='Scikit-Learn Predictions')

# Plot the ideal "Perfect Fit" line (y = x)
# If the model is 100% accurate, all green dots will fall exactly on this red line.
max_val = max(np.max(y_train), np.max(predictions))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Fit', linewidth=2)

plt.title("Actual vs Predicted (Scikit-Learn Polynomial Regression)")
plt.xlabel("Actual Execution Time (ms)")
plt.ylabel("Predicted Execution Time (ms)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()