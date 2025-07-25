# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import torch as to
import numpy as np

import numpy.typing as npt
from torch import Tensor

from emmi import Gaussian

from GSSL.utils.utils import Timer, init_means


class DDGL(Gaussian):
    """
    Data Distribution Based Graph Learning.

    The Data Distribution Based Graph Learning (DDGL) algorithm described in [1].

    Parameters
    ----------
    C : int
        Number of components.
    D : int
        Dimensionality of the data.
    H : int, optional
        Dimensionality of the factors. Defaults to 5.
    init_prior : torch.Tensor, npt.NDArray or "flat", optional
        Initial values for the priors of the mixture components. Defaults to "flat", which initializes flat priors `1/C`.
    init_means : torch.Tensor, npt.NDArray or {"kmeanspp", "random"}, optional
        Initial values for the means of the mixture components. "kmeanspp": k-means++ initialization."random": randomly selected data point. Defaults to "kmeanspp".
    init_variance : torch.Tensor, npt.NDArray or "data_variance", optional
        Initial values for the diagonal variance. Defaults to "data_variance", which uses the variance of the data.
    reg_covar : float, optional
        Regularization strength for the covariance matrix. Defaults to 1e-6.
    dtype : torch.dtype, optional
        Data type of the model parameters. Defaults to torch.float64.
    device : torch.device, optional
        Device on which the model parameters should be allocated. Defaults to CPU.

    Attributes
    ----------
    TODO
    time : dict
        Dictionary that stores runtime information for different steps.
    nnz : dict
        Dictionary that stores the number of non-zero entries of various matrices.

    References
    ----------
    [1] Y. Zhang, S. Ji, C. Zou, X. Zhao, S. Ying, and Y. Gao, "Graph learning on
        millions of data in seconds: Label propagation acceleration on graph using
        data distribution" TPAMI 45.2, pp. 1835-1847 (2023).
    """

    def __init__(
        self,
        C: int,
        D: int,
        init_prior: npt.NDArray | to.Tensor | str = "flat",
        init_means: npt.NDArray | to.Tensor | str = "afkmc2",
        init_variance: npt.NDArray | to.Tensor | str = "data_variance",
        reg_covar: float = 1e-6,
        dtype: to.dtype = to.float64,
        device: to.device = None,
    ) -> None:

        Gaussian.__init__(
            self,
            C,
            D,
            covariance_type="diagonal",
            init_prior=init_prior,
            init_means=init_means,
            init_variance=init_variance,
            shared=True,
            reg_covar=reg_covar,
            dtype=dtype,
            device=device,
        )

        self.Z = None
        self.W = None
        self.L = None
        self.soft_class_matrix = None

        self.time = {}
        self.nnz = {}

        self.dtype = dtype

    def _compute_W(self):
        """
        Computes weighted adjacency matrix of the graph.
        """
        self.W = to.matmul(self.Z.T, self.Z)

    def _compute_L(self):
        """
        Computes normalized Laplacian matrix of the graph.
        """
        W = self.W
        D = W.sum(dim=1)
        D[D == 0.0] = 1.0
        to.sqrt(D, out=D)
        to.div(1, D, out=D)

        self.L = W.clone()
        self.L *= D[:, None]
        self.L *= D[None, :]
        self.L *= -1
        diag_view = self.L.diagonal()
        diag_view += 1

    def _compute_Y(self, label_train, K):
        """
        Computes indices of labeled samples and label indicator matrix.
        """
        idx_l = to.nonzero(label_train != -1, as_tuple=True)[
            0
        ]  # without as_tuple=True not working!
        one_hot = to.concat(
            (to.eye(K, dtype=self.dtype), to.zeros([1, K], dtype=self.dtype)), axis=0
        )
        Y = one_hot[label_train[idx_l]]
        return Y, idx_l

    def _compute_soft_class_matrix(self, Y, idx_l, gamma=1.0):
        """
        Computes soft class matrix, mapping components to class scores.
        """
        Z, L = self.Z, self.L
        Z = Z[idx_l]
        LM = to.addmm(L, Z.T, Z, beta=gamma)

        diag_view = LM.diagonal()
        diag_view += 1e-6
        cholesky = to.linalg.cholesky(LM)
        RM = Z.T @ Y

        self.soft_class_matrix = to.cholesky_inverse(cholesky) @ RM

    def _compute_labels(self):
        prop_labels = to.matmul(self.Z, self.soft_class_matrix)
        return to.argmax(prop_labels, axis=1)

    def predict(
        self,
        X: Tensor | npt.NDArray,
        label_train: Tensor | npt.NDArray,
        K: int = None,
        gamma: int = 1.0,
        return_np: bool = True,
    ):
        """
        Performs label prediction (using dense matrices).
        Used, if no threshold or a threshold equal to 0 is set.
        """
        assert self.pjc is not None, "`fit` must be called before `predict`"
        label_train = (
            to.from_numpy(label_train).to(to.int64)
            if isinstance(label_train, np.ndarray)
            else label_train
        )
        K = K or label_train.max() + 1

        if self.Z is None:
            self.nnz["nnz_Z"] = self.C * label_train.shape[0]

            with Timer(self.time, "time_W"):
                self.Z = self.pjc
                self._compute_W()

            with Timer(self.time, "time_L"):
                self._compute_L()

            self.nnz["nnz_L"] = self.C**2

        with Timer(self.time, "time_Y"):
            Y, idx_l = self._compute_Y(label_train, K)

        with Timer(self.time, "time_A"):
            self._compute_soft_class_matrix(Y, idx_l, gamma=gamma)

        with Timer(self.time, "time_predict"):
            predicted_labels = self._compute_labels()

        predicted_labels[label_train != -1] = label_train[label_train != -1]
        return predicted_labels.numpy() if return_np else predicted_labels

    def fit_predict(
        self,
        X: Tensor | npt.NDArray,
        label_train: Tensor | npt.NDArray,
        eps: float = 1.0e-4,
        limit: int = None,
        batch_size: int = None,
        K: int | None = None,
        gamma: float = 1.0,
        rng: np.random.Generator | int | None = None,
        return_np: bool = True,
        verbose: bool = False,
    ):
        """
        Fits the model and predicts the labels for each sample.

        Convenience method; equivalent to calling `fit` followed by `predict`.

        Parameters
        ----------
        X : torch.Tensor or npt.NDArray
            The input data.
        label_train : torch.Tensor or npt.NDArray
            The training labels, with `-1` indicating unlabeled instances.
        eps : float, optional
            Convergence threshold. Defaults to 1.0e-4.
        limit : int or None, optional
            Maximum number of iterations. Defaults to None.
        batch_size : int or None, optional
            Size of mini-batches for training. Defaults to None, which uses the entire dataset.
        K : int, optional
            Number of unique labels or label classes. If None, it will be inferred from `label_train`.
        gamma : float, optional
            Regularization parameter for label propagation. Defaults to 1.0.
        rng : np.random.generator or int, optional
            Random number generator or seed. Defaults to None.
        return_np : bool, optional
            Whether to return `predicted_labels` as a np.array or torch.Tensor. Defaults to True.
        verbose : bool, optional
            Whether to print verbose output during the process. Defaults to False.

        Returns
        -------
        predicted_labels: torch.Tensor or npt.NDArray
            Array of predicted labels for all instances.
        """
        self.fit(
            X=X,
            eps=eps,
            limit=limit,
            batch_size=batch_size,
            rng=rng,
            verbose=verbose,
        )
        predicted_labels = self.predict(
            X=X,
            label_train=label_train,
            K=K,
            gamma=gamma,
            return_np=return_np,
        )

        return predicted_labels
