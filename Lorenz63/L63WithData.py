import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


def lorenz_63(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


# Model parameters
sigma = 10.0
rho = 28.0
beta = 8 / 3

# Initial conditions
x0, y0, z0 = 1.0, 1.0, 1.0
initial_state = (x0, y0, z0)

# Time span for integration
t_start, t_end = 0, 200
dt = 0.004  # Specify the time step here

# Time points at which to solve
t_eval = np.arange(t_start, t_end, dt)

# Solve the Lorenz 63 model
sol = solve_ivp(lorenz_63, (t_start, t_end), initial_state, args=(sigma, rho, beta), t_eval=t_eval, dense_output=True)

# Instead of generating new time points, use the ones at which you solved
t = t_eval
x, y, z = sol.y

# Save the results to a CSV file
data = np.column_stack((t, x, y, z))
df = pd.DataFrame(data, columns=['Time', 'X', 'Y', 'Z'])
df.to_csv('lorenz_63_data.csv', index=False)

# 2D plot for each variable against time
fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax[0].plot(t, x)
ax[0].set_ylabel("X")
ax[0].set_title("Lorenz 63 Model - X vs Time")

ax[1].plot(t, y)
ax[1].set_ylabel("Y")
ax[1].set_title("Lorenz 63 Model - Y vs Time")

ax[2].plot(t, z)
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Z")
ax[2].set_title("Lorenz 63 Model - Z vs Time")

plt.tight_layout()
plt.show()
