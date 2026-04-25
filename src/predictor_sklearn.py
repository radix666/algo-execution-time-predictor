import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '../data/algo_data.csv')
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)

x_train = data[:, :4]
y_train = data[:, 4]


poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x_train)

scaler = StandardScaler()
x_norm = scaler.fit_transform(x_poly)

model = LinearRegression()
model.fit(x_norm, y_train)

predictions = model.predict(x_norm)
mse = mean_squared_error(y_train, predictions)
print(f"Cost (MSE) using Scikit-Learn: {mse:.2f}")

x_new = np.array([[15000, 70, 2, 6]])
x_new_poly = poly.transform(x_new)      
x_new_norm = scaler.transform(x_new_poly) 
predicted_time = model.predict(x_new_norm)
print(f"Predicted Execution Time: {predicted_time[0]:.2f} ms")


plt.figure(figsize=(8, 6))
plt.scatter(y_train, predictions, color='green', alpha=0.5, label='Scikit-Learn Predictions')

max_val = max(np.max(y_train), np.max(predictions))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Fit', linewidth=2)

plt.title("Actual vs Predicted (Scikit-Learn Polynomial Regression)")
plt.xlabel("Actual Execution Time (ms)")
plt.ylabel("Predicted Execution Time (ms)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()