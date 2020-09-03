# -*- coding: utf-8 -*-


import numpy as np


def compute_PCA(X, mean=True, k=0, A=None, normalize=True):
    """
    PCA w.r.t A-norm, where A is a positive definite matrix.
    X : column-wise data set with size n-by-m, i.e. m samples of n dimension
    k : number of modes required (not include mean)
    normalize : whether to normalize eigenvectors w.r.t A
    Output :
        phi : n-by-(k+1)
        w   : eigenvalues
    """

    n, m = X.shape
    phi = np.zeros((n, k + 1))
    if mean:
        phi[:, 0] = X.mean(1)
        X = X - np.tile(phi[:, 0].reshape((-1, 1)), (1, m))
        s3 = 'Nonzero mean'
    else:
        s3 = 'Zero mean'

    if A is None:
        w, v = np.linalg.eigh(X.transpose() @ X)
        s1 = 'without norm'
    else:
        w, v = np.linalg.eigh(X.transpose() @ A @ X)
        s1 = 'with norm-A'
    w = w[::-1]
    v = v[:, ::-1]
    w = np.sqrt(np.absolute(w))
    if normalize:
        phi[:, 1:] = (X @ v[:, 0:k]) / np.tile(w[0:k].reshape((1, -1)), (n, 1))
        s2 = 'normalized eigenvectors'
    else:
        phi[:, 1:] = X @ v[:, 0:k]
        s2 = 'unnormalized eigenvectors'
    # print('Done PCA : (X %d-by-%d)' % (n, m) + s2 + ' ' + s1 + '. ' + s3 + ' + %d dominant eigenvectors' % (k))
    return phi, w
