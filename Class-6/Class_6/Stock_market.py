import numpy as np
import matplotlib.pyplot as plt
# Simulated stock prices
days = np.arange(1, 31)
prices = np.cumsum(np.random.randn(30) * 2 + 100) # Randomized stock prices
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(days, prices, linestyle='-', marker='o', color='b', markersize=6,
label="Stock Price")
# Adding labels, title, and legend
plt.xlabel("Day")
plt.ylabel("Price (USD)")
plt.title("Stock Price Trend")
plt.legend()
plt.grid(True)
# Show the plot
plt.show()


