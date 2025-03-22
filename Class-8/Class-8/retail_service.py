import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_sales = pd.read_csv("Retail_sales.csv")  # Ensure dataset exists

# Handle categorical feature (season) if needed
if data_sales['season'].dtype == 'object':
    data_sales = pd.get_dummies(data_sales, columns=['season'], drop_first=True)

# Define features and target
X_sales = data_sales[['ad_budget', 'discount_rate', 'store_traffic'] +
                     [col for col in data_sales.columns if 'season_' in col]]
y_sales = data_sales['sales']

# Split data
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales, y_sales, test_size=0.2, random_state=42
)

# Train model
model_sales = LinearRegression()
model_sales.fit(X_train_sales, y_train_sales)

# Predictions
y_pred_sales = model_sales.predict(X_test_sales)

# Evaluate model
mse_sales = mean_squared_error(y_test_sales, y_pred_sales)
r2_sales = r2_score(y_test_sales, y_pred_sales)

print(f"Sales Forecast - MSE: {mse_sales:.2f}")
print(f"Sales Forecast - R-squared: {r2_sales:.2f}")

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test_sales, y_pred_sales, alpha=0.7, color='blue')
plt.plot([y_test_sales.min(), y_test_sales.max()],
         [y_test_sales.min(), y_test_sales.max()], 'r', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
