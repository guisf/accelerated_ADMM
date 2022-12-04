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

    return error

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
    return M



if __name__ == '__main__':

    plt.style.use('seaborn-muted')
    
    # parameters ##########
    # see Table 1 of Candes paper (Robust Principal Component Analysis)
    n = 1000 
    r = int(0.05*n)
    #r = int(0.2*n)
    k = int(0.1*n**2)
    mi = 70
    tol = 1e-20
    lamb = 1.0/np.sqrt(n)
    rho = 1
    alpha = 1
    alpha_range=[0.7, 1.3]
    
    M = gen_data(n, r, k)
    X = np.zeros((n,n))
    Z = np.zeros((n,n))
    
    def run_algo(accel=None, damp=3):
        a1, a2 = alpha_range
        y1 = ADMM(M,lamb,X,Z,rho=rho,alpha=1,accel=accel,damp=damp,
                  maxiter=mi,tol=tol)
        y2 = ADMM(M,lamb,X,Z,rho=rho,alpha=a1,accel=accel,damp=damp,
                  maxiter=mi,tol=tol)
        y3 = ADMM(M,lamb,X,Z,rho=rho,alpha=a2,accel=accel,damp=damp,
                  maxiter=mi,tol=tol)
        return y1, y2, y3
    
    #y11, y12, y13 = run_algo()
    #y21, y22, y23 = run_algo(accel='nest', damp=3)
    #y31, y32, y33 = run_algo(accel='hb', damp=0.75)

    #pickle.dump([y11,y12,y13,y21,y22,y23,y31,y32,y33], 
    #                open('rpca.pickle','wb'))
    #pickle.dump([y11,y12,y13,y21,y22,y23,y31,y32,y33], 
    #                open('rpca2.pickle','wb'))

    #y11,y12,y13,y21,y22,y23,y31,y32,y33 = pickle.load(open('rpca.pickle','rb'))
    y11,y12,y13,y21,y22,y23,y31,y32,y33 = pickle.load(open('rpca2.pickle','rb'))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    x = range(len(y11))
    ax.plot(x, y11, marker='o', markevery=0.1, markeredgecolor='w', 
            label='R-ADMM')
    ax.fill_between(x, y12, y13, alpha=.3)
    ax.plot(x, y21, marker='s', markevery=0.1, markeredgecolor='w',
            label='R-A-ADMM')
    #ax.fill_between(x, y22, y23, alpha=.3)
    p = ax.plot(x, y31, marker='D', markevery=0.1, markeredgecolor='w',
            label='R-HB-ADMM')
    ax.fill_between(x, y32, y33, alpha=.3, color=p[0].get_color())
    ax.set_xlabel(r'iteration')
    ax.set_ylabel(r'error (log)')
    ax.legend(loc=0)
    #fig.savefig('rpca1.pdf', bbox_inches='tight')
    fig.savefig('rpca2.pdf', bbox_inches='tight')
