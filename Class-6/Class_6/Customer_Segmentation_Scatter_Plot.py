import matplotlib.pyplot as plt
import numpy as np
# Simulated customer data
np.random.seed(42)
age = np.random.randint(18, 60, 100)
spending_score = np.random.randint(20, 100, 100)
categories = np.random.choice([1, 2, 3, 4], 100) # 4 different customer types
# Scatter plot with category-based colors
plt.figure(figsize=(8, 5))
scatter = plt.scatter(age, spending_score, c=categories, cmap='jet',
alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Analysis")
plt.colorbar(label="Customer Category")
plt.grid()
plt.show()