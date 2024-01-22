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
beta = 8/3

# Initial conditions
x0, y0, z0 = 1.0, 1.0, 1.0
initial_state = (x0, y0, z0)

# Time span for integration
t_start, t_end = 0, 200
time_span = (t_start, t_end)

# Solve the Lorenz 63 model
sol = solve_ivp(lorenz_63, time_span, initial_state, args=(sigma, rho, beta), dense_output=True)

# Generate data with Gaussian noise
t_sample = np.linspace(t_start, t_end, 10000)
x_sample, y_sample, z_sample = sol.sol(t_sample)

# Add Gaussian noise
# Standard normal distribution where mean = 0 and variance = 1
noise_std = 0.5
x_noise = np.random.normal(0, noise_std, x_sample.shape)
y_noise = np.random.normal(0, noise_std, y_sample.shape)
z_noise = np.random.normal(0, noise_std, z_sample.shape)

x_sample_noise = x_sample + x_noise
y_sample_noise = y_sample + y_noise
z_sample_noise = z_sample + z_noise

# Plot the noisy data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_sample_noise, y_sample_noise, z_sample_noise, c=t_sample, cmap='viridis', marker='.')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz 63 model with Gaussian noise")
plt.show()
