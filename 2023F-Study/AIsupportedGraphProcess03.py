import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d


def load_data(file):
    data = pd.read_csv(file)
    x = data['x'].to_numpy()
    V = data['Potential(V)'].to_numpy()
    return x, V


def compute_derivative(x, V, interval):
    dx = np.gradient(V, interval)
    return dx


def plot_data(x, y, xlabel, ylabel, title, legend_label, ylim=None):
    plt.figure()
    plt.plot(x, y, label=legend_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc='upper right')
    if ylim:
        plt.ylim(ylim)
    plt.show()


def find_peaks_and_valleys(x, dx_smoothed):
    # Find the peak indices
    peak_indices = find_peaks(dx_smoothed, distance=350)[0]
    valley_indices = find_peaks(-dx_smoothed, distance=350)[0]

    # Create list of x values and peak potentials
    peak_info = [i for i in peak_indices]

    # Create list of x values and valley potentials
    valley_info = [i for i in valley_indices]

    return peak_info, valley_info


def plot_peaks_valleys(x, V, dx_smoothed, peak_indices, valley_indices):
    plt.figure()
    plt.plot(x, dx_smoothed, label="Smoothed Derivative of Potential(V)")
    plt.scatter([x[i] for i in peak_indices], [dx_smoothed[i] for i in peak_indices], c='red', label="Peaks")
    plt.scatter([x[i] for i in valley_indices], [dx_smoothed[i] for i in valley_indices], c='blue', label="Valleys")
    plt.title('Plot of smoothed derivative of Potential(V) from LCO_OCV1.csv')
    plt.xlabel('x')
    plt.ylabel('dPotential(V)/dx')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.ylim([-0.00011, 0.001])
    plt.show()


def main():
    plot_derivative_data_from_file('LCO_OCV1.csv', 10, 2, 20)


def plot_derivative_data_from_file(filename, derivative_interval_1, derivative_interval_2, smoothing_sigma):
    x, V = load_data(filename)
    plot_derivative_data(x, V, derivative_interval_1)
    compute_and_plot_peaks_valleys(x, V, derivative_interval_2, smoothing_sigma)


def plot_derivative_data(x, V, interval):
    dV = compute_derivative(x, V, interval)
    plot_data(x, dV, 'x', 'dPotential(V)/dx', 'Plot of derivative of Potential(V) from ' + filename,
              "Derivative of Potential(V)", ylim=[-0.001, 0.001])


def compute_and_plot_peaks_valleys(x, V, interval, smoothing_sigma):
    dV = compute_derivative(x, V, interval)
    dV_smoothed = gaussian_filter1d(dV, sigma=smoothing_sigma)
    peak_indices, valley_indices = find_peaks_and_valleys(x, dV_smoothed)
    plot_peaks_valleys(x, V, dV_smoothed, peak_indices, valley_indices)
