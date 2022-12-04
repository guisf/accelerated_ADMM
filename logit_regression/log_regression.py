# Guilherme Franca <guifranca@gmail.com>, Dec 6, 2018

"""
Logistic regression with l_1 norm:

    min_w 1/n \sum_{i=1}^n \log(1+e^{-y_i w^T x_i}) + \lambda\|A w\|_1 

where y_i is the label in {-1,1} from datapoint x_i.

"""

from __future__ import division

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

plt.style.use('seaborn-muted')


# parameters
n = 100
d = 500
lamb = 4
sigma = 1
fname = 'log_reg.pdf' 
maxiter = 550

w0 = 1*np.ones(d)
p0 = np.zeros(d)
rho = 1e4
alpha_range = [0.4, 1.4]
r_nest = 3
r_hb = -np.log(0.5)


def regression_data(n, d, sigma):
    a = np.random.binomial(1, 0.5, n)
    n1, n2 = len(a[a==0]), len(a[a==1])
    X1 = np.random.multivariate_normal(1*np.ones(d), sigma*np.eye(d), n1)
    X2 = np.random.multivariate_normal(-1*np.ones(d), sigma*np.eye(d), n2)
    X = np.concatenate((X1,X2))
    y = np.concatenate(([0]*n1, [1]*n2))
    idx = np.random.choice(range(n), n, replace=False)
    return X[idx,:], y[idx]

def obj(w, X, y, lamb):
    """Objective function for the above problem."""
    n, d = X.shape
    f = 0
    for i in range(n):
        f += np.log(1+np.exp(-y[i]*w.dot(X[i])))
    #f = f/n
    f += lamb*np.linalg.norm(w, 1)
    return f

def soft(x, lamb):
    """Soft-threshold operator for L_1 norm."""
    return np.sign(x)*np.maximum(np.abs(x)-lamb, 0.0) if lamb > 0 else x

def elastic_net(x, lamb, eta):
    """Elastic net proximal operator."""
    return (1/(1+eta))*soft(x, lamb)

def timeit(f):
    """Decorator function to measure time."""
    def timed(*args, **kw):
        a = time.time()
        result = f(*args, **kw)
        b = time.time()
        #return np.linspace(0, b-a, len(result)), result
        return range(0, len(result)), result
    return timed

def aadmm(X, y, w0, lamb, rho=1, alpha=1, r=3, maxiter=1000, meth='nest'):
    """Accelerated version of ADMM, with relaxation. Also implement
    Heavy Ball accelerated variant.
    
    """
    w = w0
    z = w
    zhat = z
    u = np.zeros(len(w))
    uhat = u
    ws = [w]
    for k in range(maxiter):
        oldu = u
        oldz = z
        ###
        v = np.zeros(d)
        for i in range(n):
            e = np.exp(-y[i]*w.dot(X[i]))
            v += y[i]*e*X[i]/(1+e)
        #v = v/n
        ###
        w = zhat - uhat + 1/rho*v
        u = alpha*w + (1-alpha)*zhat + uhat
        z = soft(u, lamb/rho)
        u = u - z
        if meth=="nest": # Nesterov
            uhat = u + (k/(k+r))*(u - oldu) 
            zhat = z + (k/(k+r))*(z - oldz)
        elif meth=="hb": # Heavy Ball
            uhat = u + r*(u - oldu)
            zhat = z + r*(z - oldz)
        else:
            uhat = u
            zhat = z
        ws.append(w)
    fs = [obj(w, X, y, lamb) for w in ws]
    return fs


###############################################################################
if __name__ == '__main__':

    X, y = regression_data(n, d, sigma)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale("log")
    #ax.set_xscale("log")
    
    a = 0.5
    b = 1.5
    #aadmm(X, y, w0, lamb, rho=1, alpha=1, r=3, maxiter=1000, meth='nest'):
    f0 = aadmm(X,y,w0,lamb,rho,1,maxiter=maxiter,meth=None)
    f0min = aadmm(X,y,w0,lamb,rho,a,maxiter=maxiter,meth=None)
    f0max = aadmm(X,y,w0,lamb,rho,b,maxiter=maxiter,meth=None)
    
    f1 = aadmm(X,y,w0,lamb,rho,1,r_nest,maxiter=maxiter,meth='nest')
    f1min = aadmm(X,y,w0,lamb,rho,a,r_nest,maxiter=maxiter,meth='nest')
    f1max = aadmm(X,y,w0,lamb,rho,b,r_nest,maxiter=maxiter,meth='nest')

    f2 = aadmm(X,y,w0,lamb,rho,1,r_hb,maxiter=maxiter,meth='hb')
    f2min = aadmm(X,y,w0,lamb,rho,a,r_hb,maxiter=maxiter,meth='hb')
    f2max = aadmm(X,y,w0,lamb,rho,b,r_hb,maxiter=maxiter,meth='hb')
     
    x = range(maxiter+1)
    ax.plot(x, f0, marker='o', markevery=0.1, markeredgecolor='w', 
            label='R-ADMM')
    ax.fill_between(x, f0min, f0max, alpha=.3)

    ax.plot(x, f1, marker='s', markevery=0.1, markeredgecolor='w', 
            label='R-A-ADMM')
    ax.fill_between(x, f1min, f1max, alpha=.3)

    ax.plot(x, f2, marker='D', markevery=0.1, markeredgecolor='w', 
            label='R-HB-ADMM')
    ax.fill_between(x, f2min, f2max, alpha=.3)

    ax.legend(loc=0)
    fig.savefig(fname, bbox_inches='tight')

