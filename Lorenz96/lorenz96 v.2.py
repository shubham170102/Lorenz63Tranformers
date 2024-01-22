import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA


def lorenz_96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i - 1) % N] - x[(i + 2) % N]) * x[(i + 1) % N] - x[i] + F
    return dxdt


def visualize_lorenz_96_3d(N=40, F=8, time_span=(0, 200), initial_conditions=None, dt=0.01):
    if initial_conditions is None:
        initial_conditions = np.random.rand(N)

    # Time points at which to solve
    t_eval = np.arange(time_span[0], time_span[1], dt)

    sol = solve_ivp(lorenz_96, time_span, initial_conditions, args=(F,), t_eval=t_eval)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(sol.y.T)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Lorenz 96 Model in 3D')
    plt.show()


# Example usage:
visualize_lorenz_96_3d(N=40, F=8, time_span=(0, 10))
