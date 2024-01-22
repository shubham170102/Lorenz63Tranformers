# This is a sample Python script.
import matplotlib.pyplot
import matplotlib.pyplot as matplot
# import matplot
import numpy


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def lorenz():
    h = 0.001
    end_time = 50
    num_steps = int(end_time / h)

    sigma = 10
    beta = 8 / 3
    rho = 28

    x = numpy.zeros([num_steps + 1, 2])
    y = numpy.zeros([num_steps + 1, 2])
    z = numpy.zeros([num_steps + 1, 2])

    x[0, 0] = 0
    y[0, 0] = 0.3
    z[0, 0] = 40

    x[0, 1] = 0
    y[0, 1] = 0.300000000000001
    z[0, 1] = 40

    for step in range(num_steps):
        x[step + 1] = x[step] + h * sigma * (y[step] - x[step])
        y[step + 1] = y[step] + h * (x[step] * (rho - z[step]) - y[step])
        z[step + 1] = z[step] + h * (x[step] * y[step] - beta * z[step])

    distance = numpy.abs(x[:, 1] - x[:, 0])
    return distance


distance = lorenz()
# numpy.ma.masked_less(distance, 1e-20)
# matplot.plot(distance)

matplotlib.pyplot.plot(x[:, 0], z[:, 0])
matplotlib.pyplot.plot(x[:, 1], z[:, 1])
