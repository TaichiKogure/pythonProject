#%%

####
#### 充電カーブの微分曲線を取得し、スムージングして山と谷を検出
####


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Load the data
data = pd.read_csv('LCO_OCV1.csv')
# Assuming the data has columns 'x' and 'Potential(V)'
x = data['x'].to_numpy()
V = data['Potential(V)'].to_numpy()

# Set the interval
interval = 10

# Calculate the derivatives with set interval
dV = np.gradient(V, interval)

# Plot the original data
plt.figure()
plt.plot(x, V, label="x vs Potential(V)")
plt.title('Plot of x vs Potential(V) from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('Potential(V)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

# Plot the derivative data
plt.figure()
# No need to shorten our x array as np.gradient returns array of the same length
plt.plot(x, dV, label="Derivative of Potential(V)")
plt.title('Plot of derivative of Potential(V) from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('dPotential(V)/dx')
plt.grid(True)
plt.legend(loc='upper right')
# Set the limits of y-axis
plt.ylim([-0.001, 0.001])  # replace lower_bound and upper_bound with actual values
plt.show()

######################
#スムージングしてピーク検出
######################

from scipy.ndimage.filters import gaussian_filter1d

# Load the data
data = pd.read_csv('LCO_OCV1.csv')
x = data['x'].to_numpy()
V = data['Potential(V)'].to_numpy()

# Set the interval
interval = 2

# Calculate the derivatives with set interval
dV = np.gradient(V, interval)

# Apply the Gaussian filter for data smoothing
dV_smoothed = gaussian_filter1d(dV, sigma=20)

# Find the peak indices
peak_indices = find_peaks(dV_smoothed,distance=350)[0]
valley_indices = find_peaks(-dV_smoothed,distance=350)[0]

# Create list of x values and peak potentials
peak_info = [(x[i], V[i]) for i in peak_indices]
# Create list of x values and valley potentials
valley_info = [(x[i], V[i]) for i in valley_indices]

# Plot the smoothed derivative data
plt.figure()
plt.plot(x, dV_smoothed, label="Smoothed Derivative of Potential(V)")
plt.scatter([x[i] for i in peak_indices], [dV_smoothed[i] for i in peak_indices], c='red', label="Peaks")
plt.scatter([x[i] for i in valley_indices], [dV_smoothed[i] for i in valley_indices], c='blue', label="Valleys")
plt.title('Plot of smoothed derivative of Potential(V) from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('dPotential(V)/dx')
plt.grid(True)
plt.legend(loc='upper right')
# Set the limits of y-axis
plt.ylim([-0.00011, 0.001])  # replace lower_bound and upper_bound with actual values
plt.show()

# Print the peak info
print(peak_info)
print(valley_info)

#%%
fig = plt.figure()

ax1 = fig.subplots()
ax2 = ax1.twinx()

# Setting y-axis limits for the first axis (Potential V)
ax1.set_ylim([3, 5.4])  # replace with the actual range you desire
# Setting y-axis limits for the second axis (dV_smoothed)
ax2.set_ylim([-0.0003, 0.002])  # replace with the actual range you desire

ax1.plot(x, V, color='red')  # setting color to red
ax2.plot(x, dV_smoothed, color='gray')  # setting color to blue
ax2.scatter([x[i] for i in peak_indices], [dV_smoothed[i] for i in peak_indices], c='red', label="Peaks")
ax2.scatter([x[i] for i in valley_indices], [dV_smoothed[i] for i in valley_indices], c='blue', label="Valleys")

plt.grid(True)
plt.legend(loc='upper right')
plt.show()