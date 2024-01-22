import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

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
dt = 0.01  # Specify the time step here

# Time points at which to solve
t_eval = np.arange(t_start, t_end, dt)

# Solve the Lorenz 63 model
sol = solve_ivp(lorenz_63, (t_start, t_end), initial_state, args=(sigma, rho, beta), t_eval=t_eval, dense_output=True)

# Instead of generating new time points, use the ones at which you solved
t_sample = t_eval
x_sample, y_sample, z_sample = sol.y

# Add Gaussian noise
# Standard normal distribution where mean = 0 and variance = 1
noise_std = 0.5
x_noise = np.random.normal(0, noise_std, x_sample.shape)
y_noise = np.random.normal(0, noise_std, y_sample.shape)
z_noise = np.random.normal(0, noise_std, z_sample.shape)

x_sample_noise = x_sample + x_noise
y_sample_noise = y_sample + y_noise
z_sample_noise = z_sample + z_noise

# Save the noisy data to a file
data = np.column_stack((t_sample, x_sample_noise, y_sample_noise, z_sample_noise))
df = pd.DataFrame(data, columns=['Time', 'X_noise', 'Y_Noise', 'Z_Noise'])
df.to_csv("lorenz_63_noisy_data.csv", index=False)

# Plot the noisy data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_sample_noise, y_sample_noise, z_sample_noise, c=t_sample, cmap='viridis', marker='.')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz 63 model with Gaussian noise")
plt.show()

# Plot the 2D projections of the noisy data
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].scatter(t_sample, x_sample_noise, c=t_sample, cmap='viridis', marker='.')
axs[0].set_xlabel("Time")
axs[0].set_ylabel("X")
axs[0].set_title("X vs Time")

axs[1].scatter(t_sample, y_sample_noise, c=t_sample, cmap='viridis', marker='.')
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Y")
axs[1].set_title("Y vs Time")

axs[2].scatter(t_sample, z_sample_noise, c=t_sample, cmap='viridis', marker='.')
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Z")
axs[2].set_title("Z vs Time")

plt.tight_layout()
plt.show()

