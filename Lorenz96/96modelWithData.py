import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def lorenz_96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i - 1) % N] - x[(i + 2) % N]) * x[(i + 1) % N] - x[i] + F
    return dxdt


def generate_lorenz_96_data(N=40, F=8, initial_conditions=None, dt=0.005):
    if initial_conditions is None:
        initial_conditions = np.random.rand(N)

    # Total simulation time to get 50,000 data points
    T = dt * 50000

    # Time points at which to solve
    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(lorenz_96, (0, T), initial_conditions, args=(F,), t_eval=t_eval)

    # Convert the solution to a pandas DataFrame and save to a CSV file
    df = pd.DataFrame(sol.y.T, columns=[f'x{i}' for i in range(N)])
    df.to_csv('lorenz_96_data.csv', index=False)


# Example usage:
generate_lorenz_96_data(N=40, F=8)
