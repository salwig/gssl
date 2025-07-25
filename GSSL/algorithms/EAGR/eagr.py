# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy.typing as npt

import numpy as np

from scipy.sparse import csr_matrix, diags, identity

from GSSL.utils.anchor_embedding import FLAE
from GSSL.utils.kmeans import KMeans
from GSSL.utils.utils import Timer


class EAGR(KMeans):
    """
    Efficient Anchor Graph Regularization.

    The Efficient Anchor Graph Regularization (EARG) algorithm described in [1].

    Parameters
    ----------
    C : int
        Number of components.
    D : Ignored
        Dimensionality of the data. Not used, present for API consistency.
    init_means : np.ndarray, optional
        Initial values for the means for k-means. Defaults to vanilla k-means++ initialization.

    Attributes
    ----------
    TODO

    References
    ----------
    [1] M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu, "Scalable semi-supervised learning
        by efficient anchor graph regularization", TKDE 28.7, pp. 1864-1877 (2016).
    """

    def __init__(
        self, C: int, D: int = 0, init_means: npt.NDArray | str = "afkmc2"
    ) -> None:
        self.C = C
        self.active = C
        self.D = D
        self.init_means = init_means

        self.Z = None
        self.W = None
        self.L = None

        self.time = {}
        self.nnz = {}

    def _compute_W(self):
        """
        Computes weighted adjacency matrix of the graph.
        """
        self.W = self.Z.T @ self.Z

    def _compute_L(self):
        """
        Computes normalized Laplacian matrix of the graph.
        """
        W = self.W
        Dt = np.array(W.sum(axis=1)).flatten()
        np.sqrt(Dt, out=Dt)
        np.divide(1, Dt, out=Dt)
        Dt = diags(Dt)

        self.L = identity(W.shape[0])
        self.L -= Dt @ W @ Dt

    def _compute_Y(self, label_train: npt.NDArray):
        """
        Computes indices of labeled samples and sparse label indicator matrix.
        """
        idx_l = np.nonzero(label_train != -1)[0]
        label_size = idx_l.shape[0]
        Y = csr_matrix(
            (np.ones(label_size), (np.arange(label_size), label_train[idx_l]))
        )
        return Y, idx_l

    def _compute_soft_class_matrix(
        self, Y: csr_matrix, idx_l: npt.NDArray, gamma: float = 1.0
    ):
        """
        Computes soft class matrix, mapping components to class scores.
        """
        Zl = self.Z[idx_l]

        # Regularization
        LM = Zl.T @ Zl + gamma * self.L
        LM += diags(1e-6 * np.ones(self.C))
        RM = Zl.T @ Y

        self.soft_class_matrix = np.linalg.inv(LM.todense()) @ RM

    def predict(
        self,
        X: npt.NDArray,
        label_train: npt.NDArray,
        K: int = None,
        s: int = 3,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        Performs label prediction.

        Parameters
        ----------
        X : npt.NDArray
            The input data.
        label_train : npt.NDArray
            The training labels, with `-1` indicating unlabeled instances.
        s : int, optional
            Number of nearest anchors for each data point. Defaults to 3.
        beta : float, optional
            Regularization parameter for FLAE. Defaults to 1.0.
        gamma : float, optional
            Regularization parameter for label propagation. Defaults to 1.0.

        Returns
        -------
        predicted_labels: npt.NDArray
            Array of predicted labels for all instances.

        Note
        ----
        The `fit` method must be called before using `predict`.
        """
        if self.L is None:
            with Timer(self.time, "time_Z"):
                self.Z = FLAE(X, self.anchor, s=s, beta=beta)

            self.nnz["nnz_Z"] = self.Z.nnz

            with Timer(self.time, "time_W"):
                self._compute_W()

            with Timer(self.time, "time_L"):
                self._compute_L()

            self.nnz["nnz_L"] = self.L.nnz

        with Timer(self.time, "time_Y"):
            Y, idx_l = self._compute_Y(label_train)

        with Timer(self.time, "time_A"):
            self._compute_soft_class_matrix(Y, idx_l, gamma)

        with Timer(self.time, "time_predict"):
            prop_labels = np.array(self.Z @ self.soft_class_matrix)
            prop_labels = prop_labels @ np.diag(1 / np.sum(prop_labels, axis=0))

            predicted_labels = np.argmax(prop_labels, axis=1)

            predicted_labels[label_train != -1] = label_train[label_train != -1]
        return predicted_labels

    def fit_predict(
        self,
        X: npt.NDArray,
        label_train: npt.NDArray,
        eps: float = 1e-4,
        limit: int = 300,
        s: int = 3,
        beta: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | int | None = None,
        verbose: bool = False,
    ):
        """
        Fits the model and predicts the labels for each sample.

        Convenience method; equivalent to calling `fit` followed by `predict`.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        label_train : npt.NDArray
            The training labels, with `-1` indicating unlabeled instances.
        eps : float, optional
            Convergence tolerance. Defaults to 1e-4.
        limit : int, optional
            Maximum number of iterations. Defaults to 300.
        s : int, optional
            Number of nearest anchors for each data point. Defaults to 3.
        beta : float, optional
            Regularization parameter for FLAE. Defaults to 1.0.
        gamma : float, optional
            Regularization parameter for label propagation. Defaults to 1.0.
        rng : np.random.Generator, int or None, optional
            Random number generator or seed for initialization. Defaults to None, which uses a random seed.
        verbose : bool, optional
            Whether to print progress messages. Defaults to False.

        Returns
        -------
        predicted_labels: npt.NDArray
            Array of predicted labels for all instances.
        """
        self.fit(X=X, eps=eps, limit=limit, rng=rng, verbose=verbose)
        return self.predict(X=X, label_train=label_train, s=s, beta=beta, gamma=gamma)
