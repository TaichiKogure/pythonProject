import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

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

# %%
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

# %%
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
plt.plot(x[:-1], dV / dx, label="dx dv Potential(V)")
plt.title('Plot of derivatives of x and Potential(V) from LCO_OCV1.csv')
plt.xlabel('x')
plt.ylabel('dPotential(V)/dx')
plt.grid(True)
plt.legend(loc='upper right')

plt.show()

# %%

####
#### ちょっと修正
####


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

# Scipyで記載し直し
import numpy as np
import matplotlib.pyplot as plt

# Plot the data
plt.plot(x, dV, label="Data")  # Uncomment this line

# Degree of the polynomial
degrees = [1, 3, 5, 7, 10, 20]
for degree in degrees:
    # Fit the data using polyfit for the given degree
    parameters = np.polyfit(x, dV, degree)
    # Use the fitted parameters to create a polynomial function
    polynomial = np.poly1d(parameters)
    # Create a smooth x for plotting the polynomial
    x_smooth = np.linspace(min(x), max(x), 100)
    plt.plot(x_smooth, polynomial(x_smooth), label=f"Fit Degree {degree}")



plt.title('Fitting polynomial of different degrees')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ylim([-0.001, 0.001])  # r
plt.show()


# Peak detection for Data
peaks_data, _ = find_peaks(dV)
plt.plot(x[peaks_data], dV[peaks_data], "x")  # Plot the peaks in the data

# Peak detection for each Fit Degree
for degree in degrees:
    # Fit the data using polyfit for the given degree
    parameters = np.polyfit(x, dV, degree)
    # Use the fitted parameters to create a polynomial function
    polynomial = np.poly1d(parameters)

    # Peak detection for the fitted curve
    peaks_fit, _ = find_peaks(polynomial(x))
    plt.plot(x[peaks_fit], polynomial(x)[peaks_fit], "o")  # Plot the peaks in the fitted curve


plt.title('Fitting polynomial of different degrees')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ylim([-0.00, 0.001])  # r
plt.show()
