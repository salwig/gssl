# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy.typing as npt

import numpy as np

from scipy.sparse import csr_matrix, diags, identity

from GSSL.utils.anchor_embedding import FLAE
from GSSL.utils.kmeans import KMeans
from GSSL.utils.utils import Timer


class rFME(KMeans):
    """
    Reduced Flexible Manifold Embedding

    The reduced Flexible Manifold Embedding (r-FME) algorithm described in [1].

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
    [1] S. Qiu, F. Nie, X. Xu, C. Qing, and D. Xu, "Accelerating flexible manifold
        embedding for scalable semi-supervised learning", TCSVT 29.9, pp. 2786--2795 (2019).
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

        self.projection = None
        self.bias = None

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

    def _compute_Y(self, label_train: npt.NDArray, K: int):
        """
        Computes indices of labeled samples and sparse label indicator matrix.
        """
        N = label_train.shape[0]
        idx_l = np.nonzero(label_train != -1)[0]
        label_size = idx_l.shape[0]
        Y = csr_matrix((np.ones(label_size), (idx_l, label_train[idx_l])), shape=(N, K))
        return Y, idx_l

    def _compute_F(
        self,
        Z: csr_matrix,
        Y: csr_matrix,
        para: dict,
        class_norm: bool,
        use_testset: bool = False,
    ):
        """
        Computes prediction matrix.
        """
        anchor = self.anchor.T
        dim, m = anchor.shape

        Xc = anchor - anchor.mean(axis=1, keepdims=True)
        W = np.linalg.solve(Xc @ Xc.T + para["gamma"] * np.eye(dim), Xc)

        n = Y.shape[0]
        u = para["uu"] * np.ones(n)
        u[np.sum(Y, axis=1).A1 == 1] = para["ul"]
        U = diags(u)

        mu_term = (para["mu"] / m) * np.ones((m, m))

        LM = (
            self.L
            + Z.T @ U @ Z
            + para["mu"] * np.eye(m)
            - mu_term
            - para["mu"] * Xc.T @ W
        )
        RM = (Z.T @ U @ Y).todense()

        A = np.linalg.solve(LM, RM)

        F = Z @ A
        F = np.array(F)

        if use_testset:
            self.projection = W @ A
            self.projection = np.array(self.projection)

            self.bias = (A.sum(axis=0) - self.projection.T @ anchor.sum(axis=1)) / m
            self.bias = np.array(self.bias).flatten()

        if class_norm:
            F = F @ np.diag(1 / np.sum(F, axis=0))

        return F

    def predict(
        self,
        X: npt.NDArray,
        label_train: npt.NDArray,
        K: int = None,
        s: int = 3,
        beta: float = 1.0,
        ul: float = 1e9,
        uu: float = 0.0,
        mu: float = 1e-3,
        gamma: float = 1.0,
        class_norm: bool = True,
        use_testset: bool = False,
    ):
        """
        Performs label prediction.

        Parameters
        ----------
        X : npt.NDArray
            The input data.
        label_train : npt.NDArray
            The training labels, with `-1` indicating unlabeled instances.
        K : int, optional
            The number of unique labels or label classes. If None, it will be inferred from `label_train`.
        s : int, optional
            Number of nearest anchors for each data point. Defaults to 3.
        beta : float, optional
            Regularization parameter for FLAE. Defaults to 1.0.
        ul : float, optional
            Weight for labeled data points. Defaults to 1e9.
        uu : float, optional
            Weight for unlabeled data points. Defaults to 0.0.
        mu : float, optional
            Regularization parameter mu for label propagation. Defaults to 1e-3.
        gamma : float, optional
            Regularization parameter gamma for label propagation. Defaults to 1.0.
        class_norm : bool, optional
            Whether to normalize imbalanced skewed class distributions. Defaults to True.
        use_testset: bool, optional
            Whether to do calculations to be able to predict label on an unseen testset. If False, these computations are skipped. Defaults to False.

        Returns
        -------
        predicted_labels: npt.NDArray
            Array of predicted labels for all instances.

        Note
        ----
        The `fit` method must be called before using `predict`.
        """
        if K is None:
            K = np.max(label_train) + 1

        para = {"ul": ul, "uu": uu, "mu": mu, "gamma": gamma}

        if self.Z is None:
            with Timer(self.time, "time_Z"):
                self.Z = FLAE(X, self.anchor, s, beta)

            self.nnz["nnz_Z"] = self.Z.nnz

            with Timer(self.time, "time_W"):
                self._compute_W()

            with Timer(self.time, "time_L"):
                self._compute_L()

            self.nnz["nnz_L"] = self.L.nnz

        with Timer(self.time, "time_Y"):
            Y, _ = self._compute_Y(label_train, K)

        with Timer(self.time, "time_rFME"):
            F = self._compute_F(self.Z, Y, para, class_norm, use_testset)

        with Timer(self.time, "time_predict"):
            predicted_labels = np.argmax(F, axis=1)
            predicted_labels[label_train != -1] = label_train[label_train != -1]
        return predicted_labels

    def predict_test(self, X: npt.NDArray):
        """
        Performs label prediction on unseen data.

        Parameters
        ----------
        X : npt.NDArray
            The input data.

        Returns
        -------
        predicted_labels: npt.NDArray
            Array of predicted labels for all instances.

        Note
        ----
        The `predict` or `fit_predict` method must be called with `use_testset=True` before using `predict_test`.
        """
        assert (
            self.projection is not None and self.bias is not None
        ), "`predict` or `fit_predict` must be called with `use_testset=True` before using `predict_test`"

        F = X @ self.projection + self.bias[None]
        predicted_labels = np.argmax(F, axis=1)
        return predicted_labels

    def fit_predict(
        self,
        X: npt.NDArray,
        label_train: npt.NDArray,
        eps: float = 1e-4,
        limit: int = 300,
        K: int = None,
        s: int = 3,
        beta: float = 1.0,
        ul: float = 1e9,
        uu: float = 0.0,
        mu: float = 1e-3,
        gamma: float = 1.0,
        class_norm: bool = True,
        use_testset: bool = False,
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
        K : int, optional
            The number of unique labels or label classes. If None, it will be inferred from `label_train`.
        s : int, optional
            Number of nearest anchors for each data point. Defaults to 3.
        beta : float, optional
            Regularization parameter for FLAE. Defaults to 1.0
        ul : float, optional
            Weight for labeled data points. Defaults to 1e9.
        uu : float, optional
            Weight for unlabeled data points. Defaults to 0.0.
        mu : float, optional
            Regularization parameter mu for label propagation. Defaults to 1e-3.
        gamma : float, optional
            Regularization parameter gamma for label propagation. Defaults to 1.0.
        class_norm : bool, optional
            Whether to normalize imbalanced skewed class distributions. Defaults to True.
        use_testset: bool, optional
            Whether to do calculations to be able to predict label on an unseen testset. If False, these computations are skipped. Defaults to False.
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
        return self.predict(
            X=X,
            label_train=label_train,
            K=K,
            s=s,
            beta=beta,
            ul=ul,
            uu=uu,
            mu=mu,
            gamma=gamma,
            class_norm=class_norm,
            use_testset=use_testset,
        )
