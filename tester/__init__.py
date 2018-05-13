import numpy as np
import logging
import sys


def test_sigmoid(sigmoid_input):
    from helper import sigmoid
    res = sigmoid(sigmoid_input)

    return res


def test_softmax(softmax_input):
    from helper import softmax
    res = softmax(softmax_input)

    return res


def plot_sigmoid_scores(x, y, x_title="Input", y_title="Sigmoid Score"):
    from helper import line_graph

    line_graph(x, y, x_title=x_title, y_title=y_title)


def plot_softmax_scores(x, y, x_title="Input", y_title="Softmax Score"):
    from helper import line_graph

    line_graph(x, y, x_title=x_title, y_title=y_title)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(asctime)s: %(levelname)s - %(message)s")
    logging.info("Started running the tester functions ...")

    sigmoid_input = np.arange(0, 21)
    sigmoid_res = test_sigmoid(sigmoid_input)
    logging.info(sigmoid_res)

    plot_sigmoid_scores(sigmoid_input, sigmoid_res)

    softmax_input = np.arange(0, 21)
    softmax_res = test_softmax(softmax_input)
    logging.info(softmax_res)

    plot_softmax_scores(softmax_input, softmax_res)
