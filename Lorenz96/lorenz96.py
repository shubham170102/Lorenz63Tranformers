import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Lorenz 96 model with noise
def lorenz96_noise(X, t, sigma=0.5):
    N = len(X)
    F = 8.0
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (X[(i+1)%N] - X[(i-2)%N]) * X[(i-1)%N] - X[i] + F
    noise = sigma * np.random.normal(0, 1, N)
    return dxdt + noise

# Set initial condition
X0 = np.random.rand(10)

# Generate time points
t = np.linspace(0, 10, 1000)

# Solve the differential equation with noise
X = odeint(lorenz96_noise, X0, t)

# Plot the first three variables in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
plt.show()
