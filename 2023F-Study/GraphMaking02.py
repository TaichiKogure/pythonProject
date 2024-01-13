import numpy as np
import matplotlib.pyplot as plt

# data for x-axis
x = np.linspace(-2, 2, 400)

# data for y-axis for three different functions
y1 = np.exp(x)
y2 = np.power(3,x)*x
y3 = 3*np.log(x)+10

# Creating subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Plotting data on the first subplot
ax1.plot(x, y1)
ax1.set_title('Plot of exponential function: e^x')
ax1.set_xlabel('x')
ax1.set_ylabel('y = e*e^2*x')

# Plotting data on the second subplot
ax2.plot(x, y2, 'r')
ax2.set_title('Plot of power function: 2^x')
ax2.set_xlabel('x')
ax2.set_ylabel('y = 3^x^4')

# Plotting data on the third subplot
ax3.plot(x[x >= 0], y3[x >= 0], 'g')  # log is only defined for x > 0
ax3.set_title('Plot of logarithmic function: log(x)')
ax3.set_xlabel('x')
ax3.set_ylabel('y = 2*log(x)')

# show plot
plt.tight_layout()
plt.show()