import numpy as np
import copy, math
import matplotlib.pyplot as plt

x_train = np.array([
    [1000, 20, 1, 2],    # حالة 1: داتا قليلة ومخربقة شوية
    [5000, 80, 2, 5],    # حالة 2: داتا متوسطة ومخربقة بزاف
    [10000, 50, 2, 8],   # حالة 3: داتا كبيرة والبيسي مضغوط
    [500, 10, 1, 1],     # حالة 4: داتا صغيرة ساهلة
    [20000, 90, 3, 7],   # حالة 5: داتا ضخمة وخوارزمية معقدة
    [8000, 40, 2, 4],    # حالة 6
    [15000, 70, 2, 6],   # حالة 7
    [3000, 30, 1, 3],    # حالة 8
    [25000, 100, 3, 9],  # حالة 9: أسوأ حالة ممكنة (Worst Case)
    [12000, 60, 2, 5]    # حالة 10
])
y_train = np.array([15.2, 120.5, 340.1, 5.0, 850.3, 190.2, 410.8, 45.6, 1100.5, 290.4])

w_init = np.zeros(x_train.shape[1])  # shape (4,)print(w_init)
b_init = 0.
alpha = 0.1
iterations = 1000

ww = w_init
print(ww)

mu = np.mean(x_train, axis=0)
sigma = np.std(x_train, axis=0)
sigma_safe = np.where(sigma == 0, 1, sigma)
x_norm = (x_train - mu)/sigma_safe

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    cost = cost / (m * 2)
    return cost

def compute_gradient(x, y, w, b):
    m, n= x.shape
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
    w = copy.deepcopy(w_in)
    b = 0.0
    J_history = []
    log_every = max(1, math.ceil(num_iters / 10))

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
        
        if i % log_every == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:.6g}")
    
    return w, b, J_history


w, b, J_history = gradient_descent(x_norm, y_train, w_init, b_init, compute_gradient, compute_cost, alpha, iterations)

x_new = np.array([18000, 75, 2, 6])
X_norm = (x_new - mu) / sigma_safe
predicte = np.dot(X_norm, w) + b

print(f"Predicted Execution Time: {predicte:.2f} ms")
    