# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vamm import Gaussian
from .Base import VGL


class vGMMGL(Gaussian, VGL):
    """
    Variational Gaussian Mixture Model based graph learning and label propagation algorithm.

    Parameters
    ----------
    C : int
        Number of components.
    D : int
        Dimensionality of the data.
    init_prior : np.ndarray or "flat", optional
        Initial values for the priors of the mixture components. Defaults to "flat", which initializes flat priors `1/C`.
    init_means : np.ndarray or {"afkmc2", "random"}, optional
        Initial values for the means of the mixture components. "afkmc2": AF-KMC² initialization. "random": randomly selected data point. Defaults to "afkmc2".
    init_variance : np.ndarray or "data_variance", optional
        Initial values for the variance. Defaults to "data_variance", which uses the variance of the data.
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
        init_prior: npt.NDArray | str = "flat",
        init_means: npt.NDArray | str = "afkmc2",
        init_variance: npt.NDArray | str = "data_variance",
        reg_covar: float = 1e-6,
    ) -> None:
        Gaussian.__init__(
            self,
            C=C,
            D=D,
            covariance_type="diagonal",
            flat_prior=False,
            init_prior=init_prior,
            init_means=init_means,
            init_variance=init_variance,
            shared=True,
            reg_covar=reg_covar,
        )
        VGL.__init__(self)
        self.default_T = np.sqrt(D)
        self.use_pretrainer = False
