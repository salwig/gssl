# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy.typing as npt

import torch as to
import numpy as np
from scipy.sparse import csr_matrix

import cpp


def anchor_graph(
    X: npt.NDArray,
    anchor: npt.NDArray,
    s: int = 3,
    flag: int = 1,
    cn: int = 10,
):
    """
    Anchor Graph construction using Gaussian kernel-defined weights or Local Anchor Embedding (LAE).

    Parameters:
    -----------
    X : npt.NDArray
        The input data.
    anchor : npt.NDArray
        Anchor matrix (means of k-means).
    s : int, optional
        Number of closest anchors to each datapoint. Defaults to 3.
    flag : int, optional
        Flag for anchor graph construction: 0 for Gaussian kernel-defined weights, 1 for LAE-optimized weights. Defaults to 1.
    cn : int, optional
        Number of iterations for LAE. Ignored if flag=0. Defaults to 10.

    Returns:
    --------
    Z : csr_matrix
        Regression weight matrix.

    """
    m, d = anchor.shape
    n = X.shape[0]

    # Compute squared distances between X and Anchor
    Dis = _to_sqdist(
        to.from_numpy(X), to.from_numpy(anchor)
    ).numpy()  # _to_sqdist is faster than cdist(X, anchor, "sqeuclidean") from scipy

    # 'to.topk' is faster than 'np.argpartition' and supports parallelization.
    val, pos = to.topk(to.from_numpy(Dis), s, dim=1, largest=False, sorted=False)
    val, pos = val.numpy(), pos.numpy()

    if flag == 0:
        # Gaussian kernel-defined weights
        sigma = np.mean(np.sqrt(val[:, -1]))
        val = np.exp(-val / sigma**2)
        val /= np.sum(val, axis=1)[:, None]
    else:
        # LAE-optimized weights
        val = np.zeros_like(pos, dtype=np.float64)
        cpp.LAE(X, anchor, pos, val, cn)

    Z = csr_matrix(
        (val.flatten(), (np.repeat(np.arange(n), s), pos.flatten())), shape=(n, m)
    )
    Z.eliminate_zeros()
    return Z


def FLAE(
    X: npt.NDArray,
    anchor: npt.NDArray,
    s: int = 3,
    beta: float = 1.0,
    verbose: bool = True,
):
    """
    Fast Local Anchor Embedding (FLAE).

    Parameters:
    -----------
    X : npt.NDArray
        The input data.
    anchor : npt.NDArray
        Anchor matrix (means of k-means).
    s : int, optional
        Number of closest anchors to each datapoint. Defaults to 3.
    beta : float, optional
        Regularization parameter for FLAE. Defaults to 1.0.
    verbose : bool, optional
        Whether to print progress messages (not used). Defaults to False.

    Returns:
    --------
    Z : csr_matrix
        Regression weight matrix.
    """
    #
    n = X.shape[0]
    m = anchor.shape[0]

    Dis = _to_sqdist(
        to.from_numpy(X), to.from_numpy(anchor)
    ).numpy()  # sqdist is faster than cdist(X, anchor, "sqeuclidean") from scipy

    # Dis = Dis_to.numpy()
    # pos = np.argpartition(Dis, s, axis=1)[:, :s]
    # pos = np.ascontiguousarray(pos)

    # 'to.topk' is faster than 'np.argpartition' and supports parallelization.
    pos = to.topk(to.from_numpy(Dis), s, dim=1, largest=False, sorted=False)[1].numpy()

    data = np.zeros((n * s))
    row = np.zeros((n * s), dtype=np.int64)
    col = np.zeros((n * s), dtype=np.int64)

    # Local weight estimation
    cpp.FLAE(X, anchor, pos, s, beta, data, row, col)

    Z = csr_matrix((data, (row, col)), shape=(n, m))
    Z.eliminate_zeros()
    return Z


def _sqdist(X: npt.NDArray, Y: npt.NDArray):
    """
    Pairwise squared Euclid distance.
    """
    X_norm = np.sum(X**2, axis=1)
    Y_norm = np.sum(Y**2, axis=1)
    Dis = np.matmul(X, -2 * Y.T)
    Dis += X_norm.reshape(-1, 1)
    Dis += Y_norm.reshape(1, -1)
    return Dis


def _to_sqdist(X: to.Tensor, Y: to.Tensor):
    """
    Pairwise squared Euclid distance.

    This function is faster than the NumPy equivalent "sqdist" and has better parallelization performance.
    """
    X_norm = to.sum(X**2, dim=1)
    Y_norm = to.sum(Y**2, dim=1)
    Dis = to.matmul(X, -2 * Y.T)
    Dis += X_norm[:, None]
    Dis += Y_norm[None, :]
    return Dis
