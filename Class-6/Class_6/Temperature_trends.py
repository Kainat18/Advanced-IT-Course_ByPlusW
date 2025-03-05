import matplotlib.pyplot as plt
import numpy as np
days = np.arange(1, 11)
temp_day = np.random.randint(25, 35, size=10)
temp_night = np.random.randint(15, 25, size=10)
plt.figure(figsize=(8, 5))
# Day temperature
plt.plot(days, temp_day, linestyle='--', marker='o', color='r',
markersize=8, label="Day Temperature")
# Night temperature
plt.plot(days, temp_night, linestyle='-', marker='s', color='b',
markersize=8, label="Night Temperature")
# Adding labels, title, and legend
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.title("Daily Temperature Variation")
plt.legend()
plt.grid()
plt.show()