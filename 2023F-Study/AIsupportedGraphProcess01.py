import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the data
data = pd.read_csv('LCO_OCV1.csv')
# Assuming the data has columns 'x' and 'Potential(V)'
x = data['x']
V = data['Potential(V)']
# Plot the data
plt.figure()
# Plot x vs Potential(V) data
plt.plot(x, V, label="x vs Potential(V)")
plt.title('Plot of x vs Potential(V) from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('Potential(V)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the data
data = pd.read_csv('LCO_OCV1.csv')
# Assuming the data has columns 'x' and 'y'
x = data['x']
y = data['Potential(V)']
# Compute the derivative using numpy gradient function
dy_dx = np.gradient(y, x)
# Remove NaN values from x and dy_dx
mask = ~np.isnan(dy_dx)
x, dy_dx = x[mask], dy_dx[mask]
# Plot the data
plt.figure()
# Plot derivative data
plt.plot(x, dy_dx, label='Derivative')
plt.title('Derivative from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('LCO_OCV1.csv')
# Assuming the data has columns 'x' and 'Potential(V)'
x = data['x'].to_numpy()
V = data['Potential(V)'].to_numpy()

# Calculate the derivatives
dx = np.diff(x)
dV = np.diff(V)

# Plot the original data
plt.figure()
# Plot x vs Potential(V) data
plt.plot(x, V, label="x vs Potential(V)")
plt.title('Plot of x vs Potential(V) from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('Potential(V)')
plt.grid(True)
plt.legend(loc='upper right')

# Plot the derivative data
plt.figure()
# We should also shorten our x array by 1 so it has the same length as our derivative arrays
plt.plot(x[:-1], dV/dx, label="dx dv Potential(V)")
plt.title('Plot of derivatives of x and Potential(V) from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('dPotential(V)/dx')
plt.grid(True)
plt.legend(loc='upper right')

plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

#%%
#近似曲線で記載
from scipy.optimize import curve_fit

# Assume we want to fit the derivative with a quadratic function
def func(x, a, b, c):
    return a * x**2 + b * x + c

popt, pcov = curve_fit(func, x, dV)

plt.figure()
plt.plot(x, dV, label="Derivative of Potential(V)")
plt.plot(x, func(x, *popt), 'r-', label="Fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt))
plt.title('Plot of derivative of Potential(V) from LCO_OCV1.csv with fitting curve')
plt.xlabel('x')
plt.ylabel('dPotential(V)/dx')
plt.grid(True)
plt.legend(loc='upper right')
# Set the limits of y-axis
plt.ylim([-0.001, 0.001])  # replace lower_bound and upper_bound with actual values
plt.show()

#%%
# Calculate the absolute value of the derivatives with set interval
dV = np.abs(np.gradient(V, interval))

# The rest of your code...

# No need to shorten our x array as np.gradient returns array of the same length
plt.plot(x, dV, label="Absolute value of derivative of Potential(V)")
plt.title('Plot of absolute value of derivative of Potential(V) from LCO_OCV1.csv')
plt.ylabel('|dPotential(V)/dx|')

# Fit the absolute value of derivative with a quadratic function
popt, pcov = curve_fit(func, x, dV)

plt.figure()
plt.plot(x, dV, label="Absolute value of derivative of Potential(V)")
plt.plot(x, func(x, *popt), 'r-', label="Fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt))
plt.title('Plot of absolute value of derivative of Potential(V) from LCO_OCV1.csv with fitting curve')
plt.ylabel('|dPotential(V)/dx|')
plt.grid(True)
plt.legend(loc='upper right')
# Set the limits of y-axis
plt.ylim([-0.00, 0.001])  # replace lower_bound and upper_bound with actual values
plt.show()