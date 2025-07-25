# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy.typing as npt
from torch import Tensor

import math
import torch as to
import numpy as np

from emmi import Gaussian

from GSSL.utils.utils import Timer


class MiMoLaP(Gaussian):
    """
    Mixture Model Label Propagation.

    The 'harmonic' and 'local and global consistency' versions of Mixture Model Label Propagation (MiMoLaP) algorithms described in [1], using diagonal and shared covariance.

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
    [1] M. Chi, X. He, and S. Yu, "Mixture model label propagation", CIKM, pp. 1889-1892 (2010).
    """

    def __init__(
        self,
        C: int,
        D: int,
        flat_prior: bool = False,
        init_prior: npt.NDArray | str = "flat",
        init_means: npt.NDArray | str = "afkmc2",
        init_variance: npt.NDArray | str = "data_variance",
        reg_covar: float = 1e-6,
        dtype: to.dtype = to.float64,
        device: to.device = None,
    ) -> None:
        Gaussian.__init__(
            self,
            C=C,
            D=D,
            covariance_type="diagonal",
            flat_prior=flat_prior,
            init_prior=init_prior,
            init_means=init_means,
            init_variance=init_variance,
            shared=True,
            reg_covar=reg_covar,
            dtype=dtype,
            device=device,
        )
        self.time = {}
        self.nnz = {}

        self.W = None
        self.S = None
        self.L = None

    def _compute_W(self, rho: float = 0.5):
        """
        Computes weighted adjacency matrix of the graph based on the Probability
        Product Kernel.
        """
        to.div(1, self.variance, out=self._prec)

        means_2 = (
            self._means_prec
        )  # reuse self._means_prec as auxiliary variable to reduce memory allocation
        to.pow(self.means, 2, out=means_2)
        to.mul(means_2, self._prec, out=means_2)
        self._means_2_prec[:] = means_2.sum(dim=1)
        to.mul(self.means, self._prec, out=self._means_prec)

        self.W = 0.5 * self._means_2_prec[:, None] + 0.5 * self._means_2_prec[None, :]
        to.addmm(self.W, self._means_prec, self.means.T, alpha=-1, out=self.W)

        self.W = to.exp(-0.5 * rho * self.W, out=self.W)

        if rho != 0.5:
            det = to.prod(self.variance[0])
            self.W *= to.pow(det, 0.5 - rho)
            self.W *= to.pow(
                to.tensor([2 * math.pi], device=self.device), (0.5 - rho) * self.D
            )
            self.W *= to.pow(to.tensor([2.0 * rho], device=self.device), -0.5 * self.D)

    def _compute_Y(self, label_train: Tensor, K: int):
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

    def _compute_S(self):
        """
        Computes transition matrix of the graph.
        """
        W = self.W
        D = W.sum(dim=1)
        to.sqrt(D, out=D)
        to.div(1, D, out=D)

        self.S = W.clone()
        self.S *= D[:, None]
        self.S *= D[None, :]

    def _compute_L(self):
        """
        Computes normalized Laplacian matrix of the graph.
        """
        W = self.W
        D = W.sum(dim=1)
        to.sqrt(D, out=D)
        to.div(1, D, out=D)

        self.L = W.clone()
        self.L *= D[:, None]
        self.L *= D[None, :]
        self.L *= -1
        diag_view = self.L.diagonal()
        diag_view += 1

    def _compute_F_LG(self, Y: Tensor, idx_l: Tensor, gamma: float = 1.0):
        """
        Computes the prediction matrix for the 'local and global consistency' version.
        """
        S = self.S.clone()

        S *= -1 / (1 + gamma)
        diag_view = S.diagonal()
        diag_view += 1

        Y_tilde = self.pjc[idx_l].T @ Y
        S += 1e-6 * to.eye(S.shape[0])
        cholesky = to.linalg.cholesky(S)
        self.F = to.cholesky_inverse(cholesky) @ Y_tilde

    def _compute_F_H(self, Y: Tensor, idx_l: Tensor, K: int):
        """
        Computes the prediction matrix for the 'harmonic' version.
        """
        self.F = to.zeros(self.C, K, dtype=self.dtype)
        Y_tilde = self.pjc[idx_l].T @ Y

        labeled_comp_indices = self.pjc[idx_l].argmax(dim=1)
        unlabeled_comp_indices = to.tensor(
            list(set(range(self.C)) - set(labeled_comp_indices.tolist())),
            dtype=to.int64,
        )
        Lnn = self.L[unlabeled_comp_indices, :][:, unlabeled_comp_indices]
        Lnm = self.L[unlabeled_comp_indices, :][:, labeled_comp_indices]

        self.F[labeled_comp_indices] = Y_tilde[labeled_comp_indices]
        if len(Lnn) > 0:
            Lnn += 1e-6 * to.eye(Lnn.shape[0])
            cholesky = to.linalg.cholesky(Lnn)
            self.F[unlabeled_comp_indices] = (
                -to.cholesky_inverse(cholesky) @ Lnm @ Y_tilde[labeled_comp_indices]
            )

    def predict(
        self,
        label_train: Tensor | npt.NDArray,
        X: npt.NDArray | to.Tensor = None,
        K: int = None,
        rho: float = 0.5,
        gamma: int = 1.0,
        mode: str = "lg",
        return_np: bool = True,
    ):
        """
        Performs label prediction.

        Parameters
        ----------
        label_train : torch.Tensor or npt.NDArray
            The training labels, with `-1` indicating unlabeled instances.
        X : torch.Tensor or npt.NDArray
            The input data. Not used, but for a unified interface.
        K : int, optional
            The number of unique labels or label classes. If None, it will be inferred from `label_train`.
        gamma : float, optional
            Regularization parameter for label propagation. Defaults to 1.0.
        mode : {"lg", "h"}
            Choose which version of MiMoLaP is used (case insensitive); "lg": 'local and global consistency', "h": 'harmonic'. Defaults to "lg".
        return_np : bool, optional
            Whether to return `predicted_labels` as a np.array or torch.Tensor. Defaults to True.

        Returns
        -------
        predicted_labels: npt.NDArray
            Array of predicted labels for all instances.

        Note
        ----
        The `fit` method must be called before using `predict`.
        """

        assert self.pjc is not None, "`fit` must be called before `predict`"
        assert mode.lower() in ("lg", "h")
        label_train = (
            to.from_numpy(label_train).to(to.int64)
            if isinstance(label_train, np.ndarray)
            else label_train
        )
        if K is None:
            K = to.max(label_train) + 1

        if self.W is None:
            with Timer(self.time, "time_W"):
                self._compute_W(rho=rho)

        with Timer(self.time, "time_Y"):
            Y, idx_l = self._compute_Y(label_train, K)

        if mode.lower() in ("lg",):
            if self.S is None:
                with Timer(self.time, "time_L"):
                    self._compute_S()

            with Timer(self.time, "time_A"):
                self._compute_F_LG(Y, idx_l, gamma)

        elif mode.lower() in ("h",):
            if self.L is None:
                with Timer(self.time, "time_L"):
                    self._compute_L()

            with Timer(self.time, "time_A"):
                self._compute_F_H(Y, idx_l, K)

        self.nnz["nnz_L"] = self.C**2  # In the case of MiMoLaP^LG, L is S

        with Timer(self.time, "time_predict"):
            prop_labels = self.pjc @ self.F
            predicted_labels = to.argmax(prop_labels, axis=1)

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
        mode: str = "lg",
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
        mode : {"lg", "h"}
            Choose which version of MiMoLaP is used (case insensitive); "lg": 'local and global consistency', "h": 'harmonic'. Defaults to "lg".
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
        label_train = (
            to.from_numpy(label_train)
            if isinstance(label_train, np.ndarray)
            else label_train
        )
        self.fit(
            X=X,
            eps=eps,
            limit=limit,
            batch_size=batch_size,
            rng=rng,
            verbose=verbose,
        )
        return self.predict(
            label_train=label_train, K=K, gamma=gamma, mode=mode, return_np=return_np
        )
