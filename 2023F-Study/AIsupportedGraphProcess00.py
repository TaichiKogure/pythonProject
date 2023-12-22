import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# Define x and y
x = np.linspace(0, 4 * np.pi, 1000)
y = np.sin(x)+x/3

# Add random noise to y
noise = np.random.normal(0, 0.5, len(y))
y = y + noise

# Compute lines of best fit
poly1 = np.poly1d(np.polyfit(x, y, 1))
poly2 = np.poly1d(np.polyfit(x, y, 5))
poly3 = np.poly1d(np.polyfit(x, y, 8))

# Create a space for x values to evaluate the polynomial
x_for_poly = np.linspace(x[0], x[-1], 500)

# Create figure
plt.figure(figsize=(10,6))

# Compute differential curves for x and y
x_diff = np.diff(x,1)
y_diff = np.diff(y,1)

# Plot original data
plt.plot(x, y, label='Original data')

# Plot polynomial fits
plt.plot(x_for_poly, poly1(x_for_poly), 'r', label='Linear fit')
plt.plot(x_for_poly, poly2(x_for_poly), 'b', label='Quadratic fit')
plt.plot(x_for_poly, poly3(x_for_poly), 'g', label='Cubic fit')

plt.title('Smooth Function with Four Maxima and Best Fit Lines and Derivatives')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid(True)
plt.show()

poly3_diff = np.polyder(poly3)
x_diff_poly3 = x.copy()
y_diff_poly3 = poly3_diff(x_diff_poly3)

plt.figure(figsize=(10,6))  # Create new figure for differential curve

# Plot differential curves for polynomial fit
plt.plot(x_diff_poly3, y_diff_poly3, 'r--', label='Differential curve of cubic fit')

plt.title('Differential curve of cubic fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print('The equation of the linear fit is: \n%s' % poly1)
print('The equation of the quadratic fit is: \n%s' % poly2)
print('The equation of the cubic fit is: \n%s' % poly3)

# Identify the indices of the peaks in the differential curve of cubic fit

indices_of_peaksy= argrelextrema(y_diff_poly3, np.greater)

# List the values of the peaks

peak_valuesy = y_diff_poly3[indices_of_peaksy]


print('List of peak values in the Differential curve of cubic fit:', peak_valuesy)