import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def lorenz96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)

    for i in range(N):
        dxdt[i] = (x[(i - 1) % N] - x[(i + 2) % N]) * x[(i + 1) % N] - x[i] + F

    return dxdt


def solve_lorenz96(N, F, T, dt, x0=None):
    if x0 is None:
        x0 = np.random.rand(N)
    print(x0)
    t_span = (100, T)
    t_eval = np.arange(100, T, dt)
    sol = solve_ivp(lorenz96, t_span, x0, args=(F,), t_eval=t_eval, method="RK45")

    return sol.t, sol.y


def plot_contour_planes(t, x, title=None):
    Y, X = np.meshgrid(t, np.arange(x.shape[0]))
    Z = x

    plt.figure(figsize=(12, 6))
    plt.contourf(X, Y, Z, cmap="viridis")
    plt.colorbar(label="Amplitude")
    plt.ylabel("Time")
    plt.xlabel("Space")
    if title:
        plt.title(title)
    plt.show()


# Parameters
N = 40  # Number of variables in the system
F = 16.0  # Forcing term
T = 120.0  # Total simulation time
dt = 0.005  # Time step

# Solve the Lorenz 96 model
t, x = solve_lorenz96(N, F, T, dt)

# Plot contour planes
plot_contour_planes(t, x, title="Lorenz 96 Model: Contour Planes")
