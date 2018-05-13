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


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(asctime)s: %(levelname)s - %(message)s")
    logging.info("Started running the tester functions ...")

    sigmoid_input = np.array([2, 3, 5, 6])
    res = test_sigmoid(sigmoid_input)
    logging.info(res)

    plot_sigmoid_scores(sigmoid_input, res)

    softmax_input = np.array([2, 3, 5, 6])
    res = test_softmax(softmax_input)
    logging.info(res)
