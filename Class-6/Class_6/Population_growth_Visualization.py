import matplotlib.pyplot as plt
import numpy as np
# Sample data
countries = ["USA", "China", "India", "Brazil", "Germany"]
pop_growth = np.array([0.8, 1.2, 1.5, 0.9, 0.5]) # Growth in percentage
population = np.array([331, 1441, 1380, 213, 83]) # Population in millions
# Scatter plot with size variation
plt.figure(figsize=(8, 5))
plt.scatter(pop_growth, population, c=population, cmap='viridis',
s=population * 2, alpha=0.6)
# Labels
plt.xlabel("Population Growth (%)")
plt.ylabel("Total Population (millions)")
plt.title("Population Growth vs. Total Population")
# Adding annotations
for i, country in enumerate(countries):
 plt.annotate(country, (pop_growth[i], population[i]), fontsize=10,
ha='right')
plt.colorbar(label="Population (millions)")
plt.grid()
plt.show()