# Guilherme Franca <guifranca@gmail.com>, Dec 8, 2018

"""
Quadratic problem in the form

\min_x 1/2 x^T M x + \lambda \| Ax \|_1

We choose the eigenvalues of M to make it convex or strongly convex.

"""

from __future__ import division

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import ortho_group


# parameters ##################################################################
n = 60
#m = 50
m = 0
#range_vals = [0, 1]
range_vals = [1, 2]
lamb=0
maxiter=3000
#fname = 'quad_convex.pdf'
fname = 'quad_sconvex.pdf'

rho = 5e3 
h = 1/np.sqrt(rho)
alpha_range = np.arange(0.5, 1.5, 0.1)
damping = 3
damping2 = 3
x0 = 5*np.ones(n)
p0 = np.zeros(n)
###############################################################################


# generating matrix A
A = np.random.randn(n,n)
U, S, Vt = np.linalg.svd(A)
S = np.diag(np.ones(n))
S[n-1,n-1] = 50
A = U.dot(S.dot(Vt))
A = np.eye(n)
# generating matrix M
a, b = range_vals
M = ortho_group.rvs(dim=n)
D = np.diag(np.concatenate((np.random.uniform(a, b, n-m), np.zeros(m))))
M = M.T.dot(D.dot(M))


def obj(M, x, A, lamb):
    """Objective function for the above problem."""
    #return 0.5*(x.dot(M.dot(x)) + lamb*np.linalg.norm(A.dot(x))**2)
    return 0.5*x.dot(M.dot(x)) + lamb*np.linalg.norm(A.dot(x), 1)

def soft(x, lamb):
    """Soft-threshold operator for L_1 norm."""
    if lamb > 0:
        return np.sign(x)*np.maximum(np.abs(x)-lamb, 0.0)
    else:
        return x

def timeit(f):
    """Decorator function to measure time."""
    def timed(*args, **kw):
        a = time.time()
        r = f(*args, **kw)
        b = time.time()
        #return np.linspace(0, b-a, len(r)), r
        return range(0, len(r)), r
    return timed

@timeit
def admm(M, x0, A, rho, alpha, lamb, maxiter=1000):
    """Standard Relaxed ADMM algorithm for the quadratic problem above."""
    At = A.T
    Q = np.linalg.inv(M + rho*At.dot(A))
    x = x0
    z = A.dot(x0)
    u = np.zeros(len(x0))
    xs = [x]
    for _ in range(maxiter):
        x = -rho*Q.dot(At.dot(-z+u))
        u = alpha*A.dot(x)+u+(1-alpha)*z
        #z = rho*(u)/(rho+lamb)
        z = soft(u, lamb/rho)
        u = u-z
        xs.append(x)
    fs = [obj(M, x, A, lamb) for x in xs]
    return fs

@timeit
def aadmm(M, x0, A, rho, alpha, lamb, damping, maxiter=1000, meth="nest"):
    """Standard Relaxed ADMM algorithm for the quadratic problem above."""
    At = A.T
    Q = np.linalg.inv(M + rho*At.dot(A))
    x = x0
    z = A.dot(x0)
    u = np.zeros(len(x0))
    zhat = z
    uhat = u
    xs = [x]
    for k in range(maxiter):
        oldu = u
        oldz = z
        x = -rho*Q.dot(At.dot(-zhat+uhat))
        u = alpha*A.dot(x)+uhat+(1-alpha)*zhat
        #z = rho*(u)/(rho+lamb)
        z = soft(u, lamb/rho)
        u = u-z
        if meth=="nest": # Nesterov
            uhat = u + (k/(k+damping))*(u - oldu)
            zhat = z + (k/(k+damping))*(z - oldz)
        else: # Heavy Ball
            uhat = u + (1-damping/np.sqrt(rho))*(u - oldu)
            zhat = z + (1-damping/np.sqrt(rho))*(z - oldz)
        xs.append(x)
    fs = [obj(M, x, A, lamb) for x in xs]
    return fs

def grad(M, A, At, lamb, x):
    # computing the gradient just with subgradient
    l1 = np.sign(A.dot(x))
    l1[l1==0]=1
    g = M.dot(x) + lamb*At.dot(l1)
    # computing using the proximal operator
    #g = M.dot(x)
    #Ax = A.dot(x)
    #g += At.dot((Ax - soft(Ax, h*lamb))/h)
    return g

@timeit
def sympeuler(M, x0, p0, h, alpha, damping, lamb, A, maxiter=1000, meth='nest'):
    """Leapfrog, momentum Verlet approach."""
    r = damping
    x = x0
    p = p0
    At = A.T
    AtA = At.dot(A)
    AtA_inv = np.linalg.inv(AtA)
    xs = [x]
    if meth!="nest": # Heavy Ball
        e = np.exp(-r*h)
    t = 0
    for k in range(maxiter):
        if meth=="nest": # Nesterov
            t = t + h
            e = np.power(1+h/t, -r)
        g = grad(M, A, At, lamb, x)
        p = e*p - h*alpha*g
        x = x + h*AtA_inv.dot(p)
        xs.append(x)
    fs = [obj(M, x, A, lamb) for x in xs]
    return fs


###############################################################################

if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale("log")

    c1 = 'tab:blue'
    c2 = 'tab:orange'
    c3 = 'tab:green'
    
    for alpha in alpha_range:
        
        xs, ys = admm(M,x0,A,rho,alpha,lamb,maxiter=maxiter)
        ax.plot(xs,ys,linestyle='-',color=c1,linewidth=0.4,alpha=1)
    
        xs, ys = aadmm(M,x0,A,rho,alpha,lamb,damping,maxiter=maxiter,
                        meth="nest")
        ax.plot(xs,ys,linestyle='-',color=c2,linewidth=0.4,alpha=1)
    
        xs, ys = aadmm(M,x0,A,rho,alpha,lamb,damping2,maxiter=maxiter,meth="hb")
        ax.plot(xs,ys,linestyle='-',color=c3,linewidth=0.4,alpha=1)

    mke = 250
    
    xs, ys = sympeuler(M,x0,p0,h,alpha_range[-1],damping2,lamb,A,
            maxiter=maxiter, meth="hb")
    ax.plot(xs,ys,marker='D',markevery=mke,linestyle='--',markersize=5,
            linewidth=0.4,color=c3)
    xs, ys = sympeuler(M,x0,p0,h,alpha_range[0],damping2,lamb,A,
            maxiter=maxiter, meth="hb")
    ax.plot(xs,ys,marker='D',markevery=mke,linestyle='--',markersize=5,
            linewidth=0.4,color=c3)
    
    xs, ys = sympeuler(M,x0,p0,h,alpha_range[-1],damping2,lamb,A,
            maxiter=maxiter, meth="nest")
    ax.plot(xs,ys,marker='o',markevery=mke,linestyle='--',markersize=5,
            linewidth=0.4,color=c2)
    xs, ys = sympeuler(M,x0,p0,h,alpha_range[0],damping2,lamb,A,
            maxiter=maxiter, meth="nest")
    ax.plot(xs,ys,marker='o',markevery=mke,linestyle='--',markersize=5,
            linewidth=0.4,color=c2)

    alpha = 1

    xs, ys = admm(M, x0, A, rho, alpha, lamb, maxiter=maxiter)
    ax.plot(xs,ys,linestyle='-',color=c1,label='R-ADMM')

    xs, ys = aadmm(M,x0,A,rho,alpha,lamb,damping,maxiter=maxiter,meth="nest")
    ax.plot(xs, ys,linestyle='-',color=c2,label='R-A-ADMM')

    xs, ys = aadmm(M,x0,A,rho,alpha,lamb,damping2,maxiter=maxiter,meth="hb")
    ax.plot(xs, ys,linestyle='-',color=c3,label='R-HB-ADMM')

    xs, ys = sympeuler(M,x0,p0,h,alpha,damping,lamb,A,maxiter=maxiter,
                    meth="nest")
    ax.plot(xs,ys,marker='o',markevery=mke,linestyle='--',color=c2,
            label='Symplectic Euler (Nesterov)',fillstyle='full')

    xs, ys = sympeuler(M,x0,p0,h,alpha,damping2,lamb,A,maxiter=maxiter,
                        meth="hb")
    ax.plot(xs,ys,marker='D',markevery=mke,linestyle='--',color=c3,
            label='Symplectic Euler (Heavy Ball)',fillstyle='full')
    

    ax.legend(ncol=1, loc=3)
    #ax.set_xlim([0,maxiter+1])
    #ax.set_ylim([1e-20,1e4])
    ax.set_xlabel(r'$k$')
    #ax.set_ylabel(r'$\log f(x_k)$')
    ax.set_ylabel(r'$\log \Phi(x_k)$')
    fig.savefig(fname, bbox_inches='tight')

