# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pandas as pd

from vamm import Gaussian
from .Base import VGL


class vMFAGL(Gaussian, VGL):
    """
    Variational Mixture of Factor Analyzers based graph learning and label propagation algorithm.

    Parameters
    ----------
    C : int
        Number of components.
    D : int
        Dimensionality of the data.
    H : int, optional
        Dimensionality of the factors. Defaults to 5.
    init_prior : np.ndarray or "flat", optional
        Initial values for the priors of the mixture components. Defaults to "flat", which initializes flat priors `1/C`.
    init_means : np.ndarray or {"afkmc2", "random"}, optional
        Initial values for the means of the mixture components. "afkmc2": AF-KMC² initialization. "random": randomly selected data point. Defaults to "afkmc2".
    init_A : np.ndarray or "random", optional
        Initial values for the factor loading matrices. Defaults to "random", which fills the factor loadings with uniform random numbers in [0, 1].
    init_variance : np.ndarray or "data_variance", optional
        Initial values for the diagonal variance. Defaults to "data_variance", which uses the variance of the data.
    reg_covar : float, optional
        Regularization strength for the covariance matrix. Defaults to 1e-6.

    Attributes
    ----------
    TODO
    """

    def __init__(
        self,
        C: int,
        D: int,
        H: int = 5,
        init_prior: npt.NDArray | str = "flat",
        init_means: npt.NDArray | str = "afkmc2",
        init_A: npt.NDArray | str = "uniform",
        init_variance: npt.NDArray | str = "data_variance",
        reg_covar: float = 1e-6,
    ) -> None:
        Gaussian.__init__(
            self,
            C=C,
            D=D,
            covariance_type="mfa",
            H=H,
            flat_prior=False,
            init_prior=init_prior,
            init_means=init_means,
            init_A=init_A,
            init_variance=init_variance,
            shared=True,
            reg_covar=reg_covar,
        )
        VGL.__init__(self)
        self.default_T = D
        self.use_pretrainer = True

    def fit(
        self,
        X: npt.NDArray,
        limit: list[int] | int | None = None,
        rng: np.random.generator | int | None = None,
        eps: list[float] | float = [1.0e-4, 1.0e-4],
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        hard: bool = False,
        sim_measure: str = "KL",
        indices: npt.NDArray | None = None,
        use_pretrainer: bool = True,
        verbose: bool = False,
    ) -> tuple[float, pd.DataFrame]:
        return Gaussian.fit(
            self,
            X=X,
            limit=limit,
            rng=rng,
            eps=eps,
            C_prime=C_prime,
            G=G,
            E=E,
            hard=hard,
            sim_measure=sim_measure,
            indices=indices,
            use_pretrainer=use_pretrainer,
            verbose=verbose,
        )
