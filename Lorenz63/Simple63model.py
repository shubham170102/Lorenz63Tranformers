import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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
time_span = (t_start, t_end)

# Solve the Lorenz 63 model
sol = solve_ivp(lorenz_63, time_span, initial_state, args=(sigma, rho, beta), dense_output=True)

# Plot the solution
t = np.linspace(t_start, t_end, 10000)
x, y, z = sol.sol(t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz 63 model")
plt.show()
