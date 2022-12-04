"""L1 trending filtering through fuzzed Lasso

Involves solving  the optimization problem
$$
\min_{X,Z} 1/2 \| x - b \|^2 + \lambda \| z \|_1    s.t.  z = Fx
$$

"""

import numpy as np
from scipy.optimize import minimize
import cvxpy as cp


def _soft(mu, x):
    """Soft-threshold operator for L_1 norm."""
    return np.sign(x)*np.maximum(np.abs(x)-mu, 0)

def _objective(x, b, F, lamb):
    return 0.5*np.linalg.norm(x-b)**2 + lamb*np.linalg.norm(F.dot(x), ord=1)

def _accelerate(X, oldX, k, rho, damp=3, method=None):
    """Return accelerated variable (or not).

    Parameters
    ----------
    X : 2D array
        current solution estimate
    oldX : 2D array
        previous solution estimate
    k : int
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
        #w = k/(k+damp)
        w = (k+rho)/(k+damp+rho)
    elif method == 'hb':
        w = 1-damp/np.sqrt(rho)
    else:
        w = 0
    return X+w*(X-oldX)

def ADMM(b, F, lamb, rho=1, alpha=1, accel=None, damp=3, 
         maxiter=1000, tol=1e-4):
    """Implementation of generalized Lasso using ADMM
    and its accelerated variants.

    Input
    -----
    A : 2D array
        regression matrix
    F : 2D array
        constraint matrix
    b : 1D array
        signal
    lamb : float
        regularization constant in front of L1 norm
    x, z : 1D array
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
    x = np.zeros(len(b))
    z = np.zeros(len(b)-2) # F is an (n-2, n) matrix
    zhat = z
    u = uhat = np.zeros(len(z))
    A = np.eye(len(x)) # there is no constraint
    
    MatInv = np.linalg.inv(A.T.dot(A) + rho*(F.T.dot(F)))
    Atb = A.T.dot(b)
    
    xs = [x]
    scores = [_objective(x, b, F, lamb)]
    
    k = 1
    converged = False
    while not converged:

        zold = z
        uold = u
        xold = x

        x = MatInv.dot(Atb + rho*F.T.dot(zhat - uhat))
        z = _soft(lamb/rho, alpha*F.dot(x) + (1-alpha)*zhat + uhat)
        u = uhat + alpha*F.dot(x) + (1-alpha)*zold - z
        uhat = _accelerate(u, uold, k, rho, damp=damp, method=accel)
        zhat = _accelerate(z, zold, k, rho, damp=damp, method=accel)

        xs.append(x)
        scores.append(_objective(x, b, F, lamb))

        err1 = np.linalg.norm(xold-x)/np.max([np.linalg.norm(xold),1])
        err2 = np.abs(scores[-1] - scores[-2])/np.max([1, scores[-2]])
        if k == maxiter or err1 <= tol or err2 <= tol:
            converged = True
        k += 1

    return x, xs, scores

def cvxpy_ltrend(y, F, lamb, tol=1e-8):
    """Solve L1 trend filtering with CVX."""
    x = cp.Variable(F.shape[1])
    gamma = cp.Parameter(nonneg=True)
    objective = cp.Minimize(cp.sum_squares(y-x) + gamma*cp.norm(F*x, 1))
    p = cp.Problem(objective)
    gamma.value = lamb
    res = p.solve(verbose=False, eps_abs=tol)
    return x.value

def gen_data(n, p, sigma, b):
    """Generate time series data."""
    x = [0]
    v = np.random.uniform(-b,b)
    for t in range(0, n-1):
        nochange = np.random.binomial(1, p)
        if not nochange:
            v = np.random.uniform(-b, b)
        x.append(x[t]+v)
    x = np.array(x)
    y = x + np.random.normal(0, sigma, size=len(x))

    # 2nd order Toeplitz matrix
    F = np.zeros((n-2, n))
    for t in range(0, n-2):
        F[t, t] = 1
        F[t, t+1] = -2
        F[t, t+2] = 1
    
    return y, x, F

