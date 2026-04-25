import numpy as np
import csv
import os

num_samples = 1000
np.random.seed(42)

array_size = np.random.randint(100, 50000, num_samples)
unsorted_pct = np.random.randint(0, 101, num_samples)
nested_loops = np.random.randint(1, 4, num_samples)
cpu_load = np.random.randint(1, 11, num_samples)

base_time = (array_size / 500) ** (1 + (nested_loops * 0.2))
execution_time = base_time + (cpu_load * 15) + (unsorted_pct * 0.5)

noise = np.random.normal(0, 25, num_samples)
execution_time = np.abs(execution_time + noise)

data = np.column_stack((array_size, unsorted_pct, nested_loops, cpu_load, execution_time))
current_dir = os.path.dirname(__file__)
save_path = os.path.join(current_dir, '../data/algo_data.csv')

with open(save_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['array_size', 'unsorted_pct', 'nested_loops', 'cpu-load', 'execution_time'])
    for row in data:
        writer.writerow([int(row[0]), int(row[1]), int(row[2]), int(row[3]), round(row[4], 2)])

print(f"Data generation complete! 1000 rows saved to {save_path}")