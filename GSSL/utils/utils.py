# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import time
import numpy as np

from sklearn.cluster import KMeans as kmeans_plusplus
from vamm.utils.init_params import afkmc2, random_data


def init_means(X, init_means, C, seed, verbose=False):
    if type(init_means) is np.ndarray:
        pass
    elif init_means == "afkmc2":
        init_means = afkmc2(X, C, rng=seed, verbose=verbose)[0]
    elif init_means == "kmeanspp":
        init_means = kmeans_plusplus(
            X, n_clusters=C, random_state=seed, n_local_trials=1
        )[0]
    elif init_means == "random":
        init_means = random_data(X, C, rng=seed, verbose=verbose)
    else:
        raise RuntimeError("Initialization method for means unknown.")
    return init_means


class Timer(object):
    """
    A context manager for measuring the execution time of a code block and storing the result in a dictionary.

    Parameters
    ----------
    dictionary : dict
        A dictionary where the elapsed time will be stored.
    name : str
        The key to use in the dictionary for storing the elapsed time.
    """

    def __init__(self, dictionary, name=""):
        self.dictionary = dictionary
        self.name = name

    def __enter__(self):
        """
        Starts the timer.
        """
        self.start = time.monotonic()

    def __exit__(self, type, value, traceback):
        """
        Stops the timer and stores the elapsed time in the dictionary.
        """
        self.dictionary[self.name] = time.monotonic() - self.start
