import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x): return 1 / (1 + np.exp(-x))


def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=0)


def line_graph(x, y, x_title, y_title):
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()
