"""Principal Component Pursuit

Involves solving  the optimization problem
$$
\min_{X,Z} \| X \|_{*} + \lambda \| Z \|_1    s.t.  X + Z = M
$$

"""

from __future__ import division

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

def _svd_soft(mu, X, k=None):
    """Soft threshold SVD operator.

    Parameters
    ----------
    mu : float
        constant to threshold
    X : 2D array
        Input matrix to compute SVD
    k : int
        maximum rank for SVD computation

    Output
    ------
    Xs : 2D array
        array containing the softhreshold operation
    rank : int
        we also return the estimated rank

    """
    if k == None:
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    else:
        U, sigma, Vt = svds(X, k=k)
    sigma = np.maximum(sigma-mu, 0)
    #rank = len(sigma[sigma>0])
    return U.dot(np.diag(sigma).dot(Vt))

def _soft(mu, x):
    """Soft-threshold operator for L_1 norm."""
    return np.sign(x)*np.maximum(np.abs(x)-mu, 0)

def _sample_support(shape, m):
    """Creates random support for a matrix with uniform distribution, i.e.
    uniformly choose the location of 'm' entries.
   
    Input
    -----
    shape : list
        in the format (rows, columns)
    m : int
        number of nonzero entries

    Output
    ------
    Boolean array of dimension ``shape'' with True values on nonzero positions
    
    """
    total = shape[0]*shape[1]
    omega = np.zeros(total, dtype=int)
    ids = np.random.choice(range(total), m, replace=False)
    omega[ids] = 1
    omega = omega.reshape(shape)
    return omega

def _accelerate(X, oldX, k, rho, damp=3, method=None):
    """Return accelerated variable (or not).

    Parameters
    ----------
    X : 2D array
        current solution estimate
    oldX : 2D array
        previous solution estimate
    i : int
        current iteration
    rho : float
        penalty parameter (1/stepsize)
    method : {None, 'nest', 'hb'}
        type of acceleration
    damp : float
        damping factor value

    Output
    ------
    Xhat : 2D array
        new accelerated variable
    
    """
    if method == 'nest':
        w = k/(k+damp)
    elif method == 'hb':
        w = 1-damp/np.sqrt(rho)
    else:
        w = 0
    return X+w*(X-oldX)

def ADMM(M, lamb, X, Z, rho=1, alpha=1, accel=None, damp=3, 
         maxiter=1000, tol=1e-7):
    """Implementation of Principal Component Pursuit using ADMM
    and its accelerated variants.

    Input
    -----
    M : 2D array
        matrix that should be decomposed into sparse + low rank components
    lamb : floag
        regularization constant in front of L1 norm
    X, Z : 2D array
        initial condition
    rho : float
        penalty parameter >= 1
    alpha : float
        relaxation parameter in the range [0,2]
    accel : {None, 'nest', 'hb'}
        type of acceleration
    damp : float
        damping constant for acceleration
    maxiter : int
        maximum number of iterations
    tol : float
        tolerance for relative error
    
    """

    converged = False
    M_norm = np.linalg.norm(M)
    err = np.linalg.norm(M - X - Z)/M_norm
    error = [err]
    U = Uhat = np.zeros(M.shape)
    Zhat = Z
    
    k = 1
    while not converged:

        Zold = Z
        Uold = U

        X = _svd_soft(1/rho, M - Zhat - Uhat)
        Z = _soft(lamb/rho, alpha*M - alpha*X + (1-alpha)*Zhat - Uhat)
        U = Uhat - alpha*M + alpha*X + Z - (1-alpha)*Zhat
        Uhat = _accelerate(U, Uold, k, rho, damp=damp, method=accel)
        Zhat = _accelerate(Z, Zold, k, rho, damp=damp, method=accel)

        err = np.linalg.norm(M - X - Z)/M_norm
        error.append(err)
        if err <= tol or k == maxiter:
            converged = True
        k += 1

    return error, X, Z

def gen_data(n, r, k):
    """Generate data for robust PCA.
    
    Parameters
    ----------
    n: int
        dimension
    r: int
        rank
    k: int
        number of entries for the sparse component

    Output
    ------
    M: array
        Matrix composed of low rank + sparse
    
    """
    A = np.random.normal(0, 1/np.sqrt(n), size=(n, r))
    B = np.random.normal(0, 1/np.sqrt(n), size=(n, r))
    L_true = A.dot(B.T)
    S_true = np.random.binomial(1, 0.5, size=(n, n))
    S_true[np.where(S_true==0)] = -1
    omega = _sample_support(S_true.shape, k)
    S_true = omega*S_true
    M = L_true + S_true
    return M, L_true, S_true



if __name__ == '__main__':

    plt.style.use('seaborn-muted')
    
    n = 20 
    r = int(0.1*n)
    k = int(0.1*n**2)
    mi = 200
    tol = 1e-10
    lamb = 1.0/np.sqrt(n)
    rho = 1
    alpha = 1
    
    M, Xtrue, Ztrue = gen_data(n, r, k)
    X = np.zeros((n,n))
    Z = np.zeros((n,n))
    
    err, Xhat, Zhat = ADMM(M,lamb,X,Z,rho=rho,alpha=1,accel='hb',damp=0.75,
                           maxiter=mi,tol=tol)
    #pickle.dump([err, Xhat, Zhat, Xtrue, Ztrue], open('rpca_mat.pickle','wb'))
    #err, Xhat, Zhat, Xtrue, Ztrue = pickle.load(open('rpca_mat.pickle','rb'))

    x_err = np.linalg.norm(Xhat-Xtrue)/np.linalg.norm(Xtrue)
    z_err = np.linalg.norm(Zhat-Ztrue)/np.linalg.norm(Ztrue)
    m_err = np.linalg.norm(Xhat+Zhat-M)/np.linalg.norm(M)
    x_rank = np.linalg.matrix_rank(Xhat) 
    true_rank = np.linalg.matrix_rank(Xtrue)
    z_norm = np.linalg.norm(Zhat)
    z_true_norm = np.linalg.norm(Ztrue)
    print(m_err, x_err, z_err)
    print(x_rank, true_rank)
    print(z_norm, z_true_norm)

    fig = plt.figure(figsize=(5*3,5*2))
    
    ax = fig.add_subplot(234)
    ax.matshow(Xhat, cmap='bwr')
    ax.set_title(r'$\hat{\bm{X}}$, error=%.4f, rank=%i'%(x_err, x_rank))
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = fig.add_subplot(235)
    ax.matshow(Zhat, cmap='bwr')
    ax.set_title(r'$\hat{\bm{Z}}$, error=%.4f'%(z_err))
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = fig.add_subplot(236)
    ax.matshow(Xhat+Zhat, cmap='bwr')
    ax.set_title(r'$\hat{\bm{M}}$, error=%.4f'%(m_err))
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = fig.add_subplot(231)
    ax.matshow(Xtrue, cmap='bwr')
    ax.set_title(r'$\bm{X}$, rank=%i'%true_rank)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = fig.add_subplot(232)
    ax.matshow(Zhat, cmap='bwr')
    ax.set_title(r'$\bm{Z}$')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = fig.add_subplot(233)
    ax.matshow(M, cmap='bwr')
    ax.set_title(r'$\bm{M}$')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    
    fig.savefig('rpca_mat.pdf', bbox_inches='tight')
    #fig.savefig('rpca2.pdf', bbox_inches='tight')
