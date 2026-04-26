"""
Algorithm Execution Time Predictor - From Scratch Implementation
This script implements Multiple Linear/Polynomial Regression using Gradient Descent
to predict the execution time of an algorithm based on its characteristics.
"""

import numpy as np
import copy, math, os
import matplotlib.pyplot as plt

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
# 2. MANUAL FEATURE ENGINEERING (Polynomial)
# ==========================================
# Algorithms like sorting often have O(n^2) time complexity. 
# We manually create polynomial features to capture this non-linear relationship.
array_size_squared = (x_train[:, 0] ** 2).reshape(-1, 1)
nested_loops_squared = (x_train[:, 2] ** 2).reshape(-1, 1)

# Combine original features with the new squared features (Total: 6 features)
x_train_poly = np.hstack((x_train, array_size_squared, nested_loops_squared))

# ==========================================
# 3. INITIALIZATION & HYPERPARAMETERS
# ==========================================
w_init = np.zeros(x_train_poly.shape[1]) # Weights array matches the 6 features
b_init = 0.                              # Bias term
alpha = 0.03                             # Learning rate carefully chosen for stability
iterations = 2000                        # Number of gradient descent steps

# ==========================================
# 4. FEATURE SCALING (Z-Score Normalization)
# ==========================================
# Squaring features creates massive values. Normalization ensures all features 
# are on a similar scale, preventing Gradient Descent from diverging.
mu = np.mean(x_train_poly, axis=0)
sigma = np.std(x_train_poly, axis=0)
sigma_safe = np.where(sigma == 0, 1, sigma) # Prevent division by zero
x_norm = (x_train_poly - mu)/sigma_safe


# ==========================================
# 5. CORE MACHINE LEARNING FUNCTIONS
# ==========================================

def compute_cost(x, y, w, b):
    """Computes the Mean Squared Error (MSE) Cost Function."""
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    cost = cost / (m * 2)
    return cost

def compute_gradient(x, y, w, b):
    """Calculates the gradients (partial derivatives) for weights and bias."""
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, compute_gradient, compute_cost, alpha, num_iters):
    """Performs Gradient Descent to minimize the cost function by updating w and b."""
    w = copy.deepcopy(w_in)
    b = 0.0
    J_history = []
    log_every = max(1, math.ceil(num_iters / 10))

    for i in range(num_iters):
        # Calculate gradients and update parameters simultaneously
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost history to check for convergence
        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
        
        # Print progress
        if i % log_every == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:.6g}")
    
    return w, b, J_history

# Run the training process
w, b, J_history = gradient_descent(x_norm, y_train, w_init, b_init, compute_gradient, compute_cost, alpha, iterations)


# ==========================================
# 6. MAKE A PREDICTION ON NEW DATA
# ==========================================
# Original new data: [array_size, unsorted_pct, nested_loops, cpu_load]
x_new = np.array([15000, 70, 2, 6])

# Apply the EXACT same Feature Engineering and Normalization to the new data
x_new_poly = np.array([15000, 70, 2, 6, 15000**2, 2**2])
X_norm = (x_new_poly - mu) / sigma_safe

predicte = np.dot(X_norm, w) + b
print(f"\nPredicted Execution Time: {predicte:.2f} ms\n")


# ==========================================
# 7. DATA VISUALIZATION
# ==========================================
# Calculate predictions for the entire training set to plot them
predictions = np.dot(x_norm, w) + b 

plt.figure(figsize=(12, 5))

# Plot 1: Learning Curve (Checking if Gradient Descent converged)
plt.subplot(1, 2, 1)
plt.plot(J_history, color='blue', linewidth=2)
plt.title("Learning Curve (Gradient Descent)")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Actual vs Predicted (Checking for Underfitting/Overfitting)
plt.subplot(1, 2, 2)
plt.scatter(y_train, predictions, color='purple', alpha=0.5, label='Predictions')

# Plot the ideal "Perfect Fit" line
max_val = max(np.max(y_train), np.max(predictions))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Fit')

plt.title("Actual vs Predicted Execution Time")
plt.xlabel("Actual Execution Time (ms)")
plt.ylabel("Predicted Execution Time (ms)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()