# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import cppVGL as cpp

from GSSL.utils.utils import Timer


class VGL:
    """
    Base class for variational graph learning and label propagation algorithms.

    Attributes
    ----------
    TODO
    time : dict
        Dictionary that stores runtime information for different steps.
    nnz : dict
        Dictionary that stores the number of non-zero entries of various matrices.
    """

    def __init__(self):
        self.cppVGL = None
        self.construct_graph = True
        self.time = {}
        self.nnz = {}

    @property
    def Z(self):
        """
        the transformation matrix `Z`.
        """
        return self.em.q

    def predict(
        self,
        X: npt.NDArray,
        label_train: npt.NDArray,
        label_size: int | None = None,
        K: int | None = None,
        T: float | None = None,
        gamma: float = 1.0,
        cg_rtol: float = 1e-6,
        cg_maxiter: int = 20,
        class_norm: bool = True,
    ):
        """
        Performs label prediction for each sample.

        Parameters
        ----------
        X : npt.NDArray
            The input data.
        label_train : npt.NDArray
            The training labels, with `-1` indicating unlabeled instances.
        label_size : int, optional
            The number of labeled samples. If None, it will be inferred from `label_train`.
        C_prime : int, optional
            The number of unique labels or label classes. If None, it will be inferred from `label_train`.
        T : float, optional
            Temperature. Defaults to a temperature based in the data dimensionality.
        gamma : float, optional
            Regularization parameter for label propagation. Defaults to 1.0.
        cg_rtol : float, optional
            Relative tolerance for conjugate gradient. Defaults to 1e-6.
        cg_maxiter : int, optional
            Maximum number of iterations for conjugate gradient. Defaults to 20.
        class_norm : bool, optional
            Whether to normalize imbalanced skewed class distributions. Defaults to True.

        Returns
        -------
        predicted_labels: npt.NDArray
            Array of predicted labels for all instances.

        Note
        ----
        The `fit` method must be called before using `predict`.
        """
        assert self.em is not None, "`fit` must be called before `predict`"

        T = self.default_T if T is None else T

        label_size = label_size or np.count_nonzero(label_train + 1)
        K = K or label_train.max() + 1

        if self.cppVGL is None:
            self.cppVGL = cpp.VGL(self.em, self.C, self.active, K, self.mask, 1e-4)

        if self.construct_graph:
            self.construct_graph = False

            with Timer(self.time, "time_hot_posterior"):
                self.em.E_step(X, self, False, 1 / T)

            self.nnz["nnz_Z"] = self.em.q.nnz

            with Timer(self.time, "time_W"):
                self.cppVGL.compute_W(False)

            with Timer(self.time, "time_L"):
                self.cppVGL.compute_L()

            self.nnz["nnz_L"] = self.cppVGL.L.nnz

        with Timer(self.time, "time_A"):
            self.cppVGL.compute_soft_class_matrix(
                labels=label_train,
                label_size=label_size,
                gamma=gamma,
                cg_rtol=cg_rtol,
                cg_maxiter=cg_maxiter,
            )
        N = label_train.shape[0]
        with Timer(self.time, "time_predict"):
            prop_labels = np.zeros([N, K], dtype=np.float64)
            predicted_labels = np.zeros(N, dtype=np.uint64)
            self.cppVGL.predict(prop_labels, predicted_labels, class_norm=class_norm)

        predicted_labels[label_train != -1] = label_train[label_train != -1]
        return predicted_labels

    def fit_predict(
        self,
        X: npt.NDArray,
        label_train: npt.NDArray,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        eps: list[float] | float = [1.0e-4, 1.0e-4],
        limit: list[int] | int | None = None,
        label_size: int | None = None,
        K: int | None = None,
        T: float | None = None,
        gamma: float = 1.0,
        cg_rtol: float = 1e-6,
        cg_maxiter: int = 20,
        class_norm: bool = True,
        indices: npt.NDArray | None = None,
        rng: np.random.generator | int | None = None,
        verbose: bool = False,
    ):
        """
        Fits the model and predicts the labels for each sample.

        Convenience method; equivalent to calling `fit` followed by `predict`.

        Parameters
        ----------
        X : npt.NDArray
            The input data.
        label_train : npt.NDArray
            The array of training labels, with `-1` indicating unlabeled instances.
        C_prime : int, optional
            Number of non-zeros in truncated posterior for variational EM training. Defaults to 3.
        G : int, optional
            Component neighborhood size.. Defaults to 15.
        E : int, optional
           Number of randomly added components. Defaults to 1.
        eps : list[float] or float, optional
            Convergence threshold(s). If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. Defaults to [1.0e-4, 1.0e-4].
        limit : list[int], int, optional
            Limit for the number of iterations. If a single value is provided, it is applied to both warm-up
            and EM iterations. If a list of two values is provided, the first value is used for
            warm-up iterations and the second value for EM iterations. Defaults to None.
        label_size : int, optional
            Number of labeled samples. If None, it will be inferred from `label_train`.
        K : int, optional
            Number of unique labels or label classes. If None, it will be inferred from `label_train`.
        T : float, optional
            Temperature. Defaults to a temperature based in the data dimensionality.
        gamma : float, optional
            Regularization parameter for label propagation. Defaults to 1.0.
        cg_rtol : float, optional
            Relative tolerance for conjugate gradient. Defaults to 1e-6.
        cg_maxiter : int, optional
            Maximum number of iterations for conjugate gradient. Defaults to 20.
        class_norm : bool, optional
            Whether to normalize imbalanced skewed class distributions. Defaults to True.
        indices : npt.NDArray, optional
            Indices of data points uses as seeds. Used for initializing the K-sets and component
            neighborhood. Defaults to None.
        rng : np.random.generator or int, optional
            Random number generator or seed. Defaults to None.
        verbose : bool, optional
            Whether to print verbose output during the process. Defaults to False.

        Returns
        -------
        predicted_labels: npt.NDArray
            Array of predicted labels for all instances.
        """
        self.fit(
            X=X,
            limit=limit,
            rng=rng,
            eps=eps,
            C_prime=C_prime,
            G=G,
            E=E,
            indices=indices,
            use_pretrainer=self.use_pretrainer,
            verbose=verbose,
        )
        predicted_labels = self.predict(
            X=X,
            label_train=label_train,
            label_size=label_size,
            K=K,
            T=T,
            gamma=gamma,
            cg_rtol=cg_rtol,
            class_norm=class_norm,
            cg_maxiter=cg_maxiter,
        )

        return predicted_labels
