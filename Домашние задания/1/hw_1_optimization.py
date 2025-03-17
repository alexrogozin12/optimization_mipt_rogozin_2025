import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from typing import Callable, Optional
from scipy.stats import ortho_group

COLOR_RED = np.linspace(240, 166, 256) / 255.0
COLOR_GREEN = np.linspace(244, 188, 256) / 255.0
COLOR_BLUE = np.linspace(246, 203, 256) / 255.0


def plot_levels(func, xrange=None, yrange=None, levels=None):
    """
    Plotting the contour lines of the function.

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> plot_levels(oracle.func)
    """
    if xrange is None:
        xrange = [-6, 6]
    if yrange is None:
        yrange = [-5, 5]
    if levels is None:
        levels = [0, 0.25, 1, 4, 9, 16, 25]

    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    colors = np.vstack([COLOR_RED, COLOR_GREEN, COLOR_BLUE]).T
    my_cmap = ListedColormap(colors)

    _ = plt.contourf(X, Y, Z, levels=levels, cmap=my_cmap)
    CS = plt.contour(X, Y, Z, levels=levels, colors="#ABBECC")
    plt.clabel(CS, inline=1, fontsize=8, colors="#AAAEBB")
    plt.grid()


def plot_trajectory(history, fit_axis=False, label=None, color="C1"):
    """
    Plotting the trajectory of a method.
    Use after plot_levels(...).

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> [x_star, msg, history] = optimization.gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
    >> plot_levels(oracle.func)
    >> plot_trajectory(oracle.func, history['x'])
    """
    x_values, y_values = zip(*history)
    plt.plot(x_values, y_values, ".-", linewidth=1.0, ms=5.0, alpha=1.0, c=color, label=label)
    plt.legend()

    # Tries to adapt axis-ranges for the trajectory:
    if fit_axis:
        xmax, ymax = np.max(x_values), np.max(y_values)
        COEF = 1.5
        xrange = [-xmax * COEF, xmax * COEF]
        yrange = [-ymax * COEF, ymax * COEF]
        plt.xlim(xrange)
        plt.ylim(yrange)


def generate_random_2d_psd_matrix(lmin: float, lmax: float):
    phi = np.random.uniform(2 * np.pi)
    s = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    A = np.array([[lmin, 0], [0, lmax]])
    return np.dot(s, np.dot(A, s.T))


def generate_random_psd_matrix(dim: int, lmin: float, lmax: float):
    lambdas = np.random.uniform(lmin, lmax, dim)
    lambdas[0] = lmin
    lambdas[-1] = lmax
    A = np.diag(lambdas)
    s = ortho_group(dim)
    A = np.array([[lmin, 0], [0, lmax]])
    return np.dot(s, np.dot(A, s.T))


def armijo(phi: Callable, der_phi: Callable, c: float = 1e-4, previous_alpha: Optional[float] = None):
    alpha = 1.0 or 2.0 * previous_alpha

    phi0 = phi(0)
    der_phi0 = der_phi(0)

    while phi(alpha) > phi0 + c * alpha * der_phi0:
        alpha /= 2
    return alpha

