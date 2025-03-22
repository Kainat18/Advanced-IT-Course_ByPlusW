
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 200 rows of synthetic retail sales data
data = {
    "ad_budget": np.random.randint(1000, 50000, 200),  # Marketing budget
    "discount_rate": np.random.uniform(5, 50, 200),  # Discount percentage
    "season": np.random.choice(["Winter", "Spring", "Summer", "Fall"], 200),  # Seasonal effect
    "store_traffic": np.random.randint(100, 5000, 200),  # Number of customers visiting the store
    "sales": np.random.randint(5000, 100000, 200)  # Total sales revenue
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("Retail_sales.csv", index=False)
print("Dataset 'Retail_sales.csv' created successfully!")
