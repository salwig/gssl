# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy.typing as npt

import time
import pandas as pd

import numpy as np

from sklearn.cluster import KMeans as sk_KMeans
from .utils import init_means


class KMeans:
    """
    Base class to fit k-means to find anchor nodes.
    """

    def fit(
        self,
        X: npt.NDArray,
        eps: float = 1e-4,
        limit: int = 300,
        rng: np.random.Generator | int | None = None,
        verbose: bool = False,
    ):
        """
        Fit k-means to the input data to find anchor nodes.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        eps : float, optional
            Convergence tolerance. Defaults to 1e-4.
        limit : int, optional
            Maximum number of iterations. Defaults to 300.
        rng : np.random.Generator, int or None, optional
            Random number generator or seed for initialization. Defaults to None, which uses a random seed.
        verbose : bool, optional
            Whether to print progress messages. Defaults to False.

        Returns
        -------
        objective : float
            the final objective value, k-means inertia per data point
        log : pd.DataFrame
            a DataFrame with training history.
        """
        N = X.shape[0]
        rng = np.random.default_rng(rng)
        seed = rng.integers(low=0, high=np.iinfo(np.uint32).max)

        self.init_means = init_means(X, self.init_means, self.C, seed, verbose)

        kmeans = sk_KMeans(
            n_clusters=self.C,
            init=self.init_means,
            n_init=1,  # to fix warnings
            max_iter=limit,
            tol=eps,
            verbose=verbose,
            random_state=seed,
        )

        tic = time.monotonic()
        kmeans.fit(X)
        dt = time.monotonic() - tic

        # TODO: rename anchor to means or means to anchor
        self.anchor = kmeans.cluster_centers_
        obj = kmeans.inertia_ / X.shape[0]
        N_iter = kmeans.n_iter_

        log = pd.DataFrame(
            {
                "i": np.arange(N_iter),
                "active": self.C,
                "M_step": True,
                "objective": np.nan,
                "eval": N * self.C,
                "time": dt / N_iter,
            }
        )
        log["objective"].values[-1] = obj
        return obj, log
