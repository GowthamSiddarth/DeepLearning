import numpy as np
import logging
import sys


def test_sigmoid():
    sigmoid_input = np.array([2, 3, 5, 6])

    from helper import sigmoid
    res = sigmoid(sigmoid_input)

    logging.info(res)


def test_softmax():
    softmax_input = np.array([2, 3, 5, 6])

    from helper import softmax
    res = softmax(softmax_input)

    logging.info(res)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(asctime)s: %(levelname)s - %(message)s")
    logging.info("Started running the tester functions ...")
    test_sigmoid()
