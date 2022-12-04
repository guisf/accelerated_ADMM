# Guilherme Franca <guifranca@gmail.com>, Dec 6, 2018

"""
Elastic Net regression problem.
We test ADMM, Accelerated ADMM, symplectic integrators, etc.

We consider the problem

    min_x (1/2)\|y - Mx\|^2 + \lambda\|A x\|_1 + (\eta/2)\|A x\|^2

where y = M q + noise and the solution x = \hat{q} is an estimator for q.

"""

from __future__ import division

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

#plt.style.use('ggplot')
#print plt.style.available

###############################################################################
# parameters
n = 40
m = 50
lamb = 1
eta = 1
sigma = 0.1
fname = 'elnet2.pdf' 
#maxiter = 250
mi = 10 # maxiter

A = np.eye(n)
x0 = np.ones(n)
p0 = np.zeros(n)
rho = 10
r_nest = 3
r_hb = 10
#print(r_hb)
###############################################################################


def regression_data(m=50, n=40, sigma=0.1):
    """Generate data for the regression problem. This set up
    is the same as in Goldstein 14 and Tibshirani original elastic net paper.

    (m, n) = (rows, columns) of matrix M, sigma is the size of the Gaussian
    noise.
    
    """
    #v1 = np.random.multivariate_normal(np.zeros(m), np.eye(m))
    #v2 = np.random.multivariate_normal(2*np.ones(m), 0.5*np.eye(m))
    #v3 = np.random.multivariate_normal(3*np.ones(m), 1.5*np.eye(m))
    v1 = np.random.multivariate_normal(np.zeros(m), np.eye(m))
    v2 = np.random.multivariate_normal(np.zeros(m), np.eye(m))
    v3 = np.random.multivariate_normal(np.zeros(m), np.eye(m))
    M = np.random.normal(0, 1, (m, n))
    for i in range(0, 5):
        M[:,i] = v1 + np.random.normal(0, sigma, m)
    for i in range(5, 10):
        M[:,i] = v2 + np.random.normal(0, sigma, m)
    for i in range(10, 15):
        M[:,i] = v3 + np.random.normal(0, sigma, m)
    q = np.concatenate((3*np.ones(15), np.zeros(n-15)))
    y = M.dot(q) + np.random.normal(0, 0.1, m)
    #y = M.dot(q) + 15*np.random.normal(0, 1, m)
    return M, y, q

def obj(M, y, x, lamb, eta):
    """Objective function for the above problem."""
    n = M.shape[1]
    return 0.5*(np.linalg.norm(M.dot(x) - y)**2)+\
           lamb*np.linalg.norm(x,1)+eta/2*np.linalg.norm(x)**2

def soft(x, lamb):
    """Soft-threshold operator for L_1 norm."""
    return np.sign(x)*np.maximum(np.abs(x)-lamb, 0.0)

def elastic_net(x, lamb, eta):
    """Elastic net proximal operator."""
    return (1/(1+eta))*soft(x, lamb)

def matrices1(M, y):
    Mt = M.T
    MtM = Mt.dot(M)
    Mty = Mt.dot(y)
    return MtM, Mty

def matrices2(M, y, A):
    Mt = M.T
    MtM = Mt.dot(M)
    Mty = Mt.dot(y)
    At = A.T
    AtA = At.dot(A)
    Qinv = np.linalg.inv(MtM + rho*AtA)
    return MtM, Mty, A, At, AtA, Qinv

def matrices3(M, y, A):
    Mt = M.T
    MtM = Mt.dot(M)
    Mty = Mt.dot(y)
    At = A.T
    AtA = At.dot(A)
    AtA_inv = np.linalg.inv(AtA)
    return MtM, Mty, A, At, AtA, AtA_inv

def timeit(f):
    """Decorator function to measure time."""
    def timed(*args, **kw):
        a = time.time()
        result = f(*args, **kw)
        b = time.time()
        #return np.linspace(0, b-a, len(result)), result
        return range(0, len(result)), result
    return timed

def admm(M, y, x0, rho, alpha, lamb, eta, A, maxiter=1000):
    """Implement ADMM method. Here we allow the iterations to
    have the operator A.
    
    """
    MtM, Mty, A, At, AtA, Qinv = matrices2(M, y, A)
    x = x0
    z = A.dot(x)
    u = np.zeros(len(x0))
    xs = [x]
    for k in range(maxiter):
        x = Qinv.dot(Mty + rho*At.dot(z-u))
        v = alpha*A.dot(x) + (1-alpha)*z + u
        z = elastic_net(v, lamb/rho, eta/rho)
        u = v - z
        xs.append(x)
    fs = [obj(M, y, x, lamb, eta) for x in xs]
    return np.array(fs), np.array(xs)

def aadmm(M, y, x0, rho, alpha, r, lamb, eta, A, maxiter=1000, meth='nest'):
    """Accelerated version of ADMM, with relaxation. Also implement
    Heavy Ball accelerated variant.
    
    """
    MtM, Mty, A, At, AtA, Qinv = matrices2(M, y, A)
    x = x0
    z = x
    zhat = z
    u = np.zeros(len(x0))
    uhat = u
    xs = [x]
    for k in range(maxiter):
        oldu = u
        oldz = z
        x = Qinv.dot(Mty + rho*At.dot(zhat-uhat))
        v = alpha*A.dot(x) + (1-alpha)*zhat + uhat
        z = elastic_net(v, lamb/rho, eta/rho)
        u = v - z
        if meth=="nest": # Nesterov
            uhat = u + (k/(k+r))*(u - oldu) 
            zhat = z + (k/(k+r))*(z - oldz)
        else: # Heavy Ball
            uhat = u + (1-r/np.sqrt(rho))*(u - oldu)
            zhat = z + (1-r/np.sqrt(rho))*(z - oldz)
        xs.append(x)
    fs = [obj(M, y, x, lamb, eta) for x in xs]
    return np.array(fs), np.array(xs)


###############################################################################
if __name__ == '__main__':
    import sys
    
    M, y, x_star = regression_data(m, n, sigma)
    obj_star = obj(M, y, x_star, lamb, eta)
    print(obj_star)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_yscale("log")

    c1 = 'tab:blue'
    c2 = 'tab:orange'
    c3 = 'tab:green'
 
    alpha = 1
    
    fs, xs = aadmm(M,y,x0,rho,alpha,r_nest,lamb,eta,A,maxiter=mi,meth='nest')
    #ys = [np.log(np.abs((f - fstar)/fstar)) for f in fs]
    ax.plot(range(len(fs)),fs,linestyle='-',color=c2,label='R-A-ADMM')

    fs, xs = aadmm(M,y,x0,rho,alpha,r_hb,lamb,eta,A,maxiter=mi,meth='hb')
    ax.plot(range(len(fs)),fs,linestyle='-',color=c3,label='R-HB-ADMM')

    ax.legend(ncol=1, loc=3)
    ax.set_xlabel(r'$k$')
    #ax.set_ylabel(r'$\log|(\Phi_k-\Phi_{\infty})/\Phi_{\infty}|$')
    #plt.yticks([4, 2, 0,-2,-4,-6,-8, -10, -12], 
    #           [r'$10^{4}$', r'$10^{2}$',r'$10^0$',r'$10^{-2}$',
    #            r'$10^{-4}$',r'$10^{-6}$',r'$10^{-8}$',
    #            r'$10^{-10}$', r'$10^{-12}$'])
    #ax.set_ylim([-10,5])
    fig.savefig(fname, bbox_inches='tight')


