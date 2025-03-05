import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Sample sales data
np.random.seed(42)
region = np.random.randint(1, 6, 50) # 5 regions
sales = np.random.randint(500, 10000, 50)
profit = np.random.randint(50, 1000, 50)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot
scatter = ax.scatter(region, sales, profit, c=profit, cmap='coolwarm', s=50,
alpha=0.7)
# Labels
ax.set_xlabel("Region")
ax.set_ylabel("Sales")
ax.set_zlabel("Profit")
ax.set_title("3D Sales Performance Analysis")
fig.colorbar(scatter, label="Profit")
plt.show()