import matplotlib.pyplot as plt
import numpy as np  # Not needed


def rearrangeticks(xmin, xmax, ymin, ymax, xstep, ystep):
    gridx = np.arange(xmin, xmax + 1, xmax)
    gridy = np.arange(ymin, ymax + 1, ymax)
    for foo in np.linspace(ymin, ymax, ystep + 1):
        plt.plot(gridx, [foo] * len(gridx), color="black", alpha=.1)
    for foo in np.linspace(xmin, xmax, xstep + 1):
        plt.plot([foo] * len(gridy), gridy, color="black", alpha=.1)
    for foo in np.linspace(ymin, ymax, ystep * 5 + 1):
        plt.plot(gridx, [foo] * len(gridx), color="black", alpha=.05)
    for foo in np.linspace(xmin, xmax, xstep * 5 + 1):
        plt.plot([foo] * len(gridy), gridy, color="black", alpha=.05)
    plt.xticks(np.linspace(xmin, xmax, xstep + 1))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
