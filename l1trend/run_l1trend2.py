"""Test the l1 trend filtering problem with variants of ADMM."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import sys
import seaborn as sns

import l1trend

plt.style.use('seaborn-muted')

def time_series(y, F, x_true, lamb,
                rho=10, alpha=1, gamma=0.99, mi=200, tol=1e-10):
    """Make the plot of recoverred signal."""
    #y, x_true, F = l1trend.gen_data(n, p, sigma, b)
    def rel_err(x):
        return np.linalg.norm(x-x_true)/np.linalg.norm(x_true)
    def err(xs):
        return [rel_err(x) for x in xs]
    x_cvx = l1trend.cvxpy_ltrend(y, F, lamb, tol=tol)
    x1, xs1, sc1 = l1trend.ADMM(y,F,lamb,rho=rho,alpha=alpha,accel=None,
                                maxiter=mi,tol=tol)
    x2, xs2, sc2 = l1trend.ADMM(y,F,lamb,rho=rho,alpha=alpha,accel='nest',
                                damp=3,maxiter=mi,tol=tol)
    x3, xs3, sc3 = l1trend.ADMM(y,F,lamb,rho=rho,alpha=alpha,accel='hb',
                                damp=gamma,
                                maxiter=mi,tol=tol)
    e0 = rel_err(x_cvx)
    e1 = rel_err(x1); k1 = len(xs1);
    e2 = rel_err(x2); k2 = len(xs2);
    e3 = rel_err(x3); k3 = len(xs3);
    print('CVX:', e0)
    print('ADMM:', e1, k1)
    print('A-ADMM:', e2, k2)
    print('HB-ADMM:', e3, k3)
    
    fig, axes = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=False,
                    #figsize=(1.3*6.4,1.3*4.8)
                    figsize=(2*6.4,2*4.8)
                    )
    #size = fig.get_size_inches()
    ax = axes.flat
   
    ax[0].plot(y, color='r', lw=1, label=r'observed signal $y$') 
    ax[0].plot(x_true, color='k', linestyle='--', label=r'underlying trend $x$')
    ax[0].legend(loc=0)
    
    ax[1].plot(x_true, color='k', linestyle='--') 
    ax[1].plot(x1, label='R-ADMM')
    ax[1].set_title('error=%.3f, iterations=%i'%(e1, k1))
    ax[1].legend(loc=0)
    
    ax[2].plot(x_true, color='k', linestyle='--') 
    ax[2].plot(x2, label='R-A-ADMM')
    ax[2].set_title('error=%.3f, iterations=%i'%(e2, k2))
    ax[2].legend(loc=0)
    
    ax[3].plot(x_true, color='k', linestyle='--') 
    ax[3].plot(x3, label='R-HB-ADMM')
    ax[3].set_title('error=%.3f, iterations=%i'%(e3, k3))
    ax[3].legend(loc=0)
    
    ax[2].set_ylabel(r'signal')
    ax[0].set_ylabel(r'signal')
    ax[2].set_xlabel(r'time')
    ax[3].set_xlabel(r'time')
    
    plt.subplots_adjust(wspace=0.15,hspace=0.15)
    #fig.savefig('l1trending.pdf', bbox_inches='tight')
    fig.savefig('l1trending2.pdf', bbox_inches='tight')

def alpha_convergence2(Y, F, x_true, lamb, 
                rho=10, gamma=0.99, mi=200, amin=0.5, amax=1.5, tol=1e-10):
    """Convergence plot with different values of alpha.
    Make sure to choose a small enough tolerance and appropriate 'mi'
    so that all trials have the same number of iterations.
    
    """
    
    #y, x_true, F = l1trend.gen_data(n, p, sigma, b)
    
    def err(xhat, x_true):
        return np.linalg.norm(xhat-x_true)/np.linalg.norm(x_true)
    
    def admm(accel=None,damp=None):
        x1, xs1, _ = l1trend.ADMM(y,F,lamb,rho=rho,alpha=amin,
                                  accel=accel,damp=damp,maxiter=mi,tol=tol)
        x2, xs2, _ = l1trend.ADMM(y,F,lamb,rho=rho,alpha=amax,
                                  accel=accel,damp=damp,maxiter=mi,tol=tol)
        x3, xs3, _ = l1trend.ADMM(y,F,lamb,rho=rho,alpha=1,
                                  accel=accel,damp=damp,maxiter=mi,tol=tol)
        ys1 = [err(x, x_true) for x in xs1]
        ys2 = [err(x, x_true) for x in xs2]
        ys3 = [err(x, x_true) for x in xs3]
        return ys1[1:], ys2[1:], ys3[1:]
        
    cvx_sol = l1trend.cvxpy_ltrend(y, F, lamb, tol=tol)
    err_cvx = err(cvx_sol, x_true)
    print('\ncvx:', err_cvx)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    xsmin, xsmax, xs = admm(accel=None)
    ax.plot(range(len(xs)), xs, marker='o', markeredgecolor='w', markevery=0.1, 
            label=r'R-ADMM')
    #print(len(xsmin), len(xsmax))
    #ax.fill_between(range(len(xmin), xsmin, xsmax, alpha=0.3)
    xsmin, xsmax, xs = admm(accel='nest',damp=3)
    ax.plot(range(len(xs)), xs, marker='s', markeredgecolor='w', markevery=0.1, 
            label=r'R-A-ADMM')
    #ax.fill_between(x, xsmin, xsmax, alpha=0.3)
    xsmin, xsmax, xs = admm(accel='hb',damp=gamma)
    ax.plot(range(len(xs)), xs, marker='D', markeredgecolor='w', markevery=0.1, 
            label=r'R-HB-ADMM')
    #ax.fill_between(x, xsmin, xsmax, alpha=0.3)
    ax.set_xlabel('iteration (log)')
    ax.set_ylabel('error (log)')
    ax.legend(loc=0)
    fig.savefig('l1trend_alpha2.pdf', bbox_inches='tight')

def many_trials(n, p, sigma, b, lamb, num_exp,
                rho=10, gamma=0.99, mi=200, tol=1e-10):
    
    def err(xhat, x_true):
        return np.linalg.norm(xhat-x_true)/np.linalg.norm(x_true)
    
    data = []
    for e in range(num_exp):

        y, x_true, F = l1trend.gen_data(n, p, sigma, b)
    
        x1, xs1, _ = l1trend.ADMM(y,F,lamb,rho=rho,alpha=1,
                                accel=None,maxiter=mi,tol=tol)
        x2, xs2, _ = l1trend.ADMM(y,F,lamb,rho=rho,alpha=1,
                                  accel='nest',damp=3,maxiter=mi,tol=tol)
        x3, xs3, _ = l1trend.ADMM(y,F,lamb,rho=rho,alpha=1,
                                  accel='hb',damp=gamma,maxiter=mi,tol=tol)
        x4 = l1trend.cvxpy_ltrend(y, F, lamb, tol=tol)
        
        e1 = err(x1, x_true)
        e2 = err(x2, x_true)
        e3 = err(x3, x_true)
        e4 = err(x4, x_true)
        
        k1 = len(xs1)
        k2 = len(xs2)
        k3 = len(xs3)

        data.append(['R-ADMM', k1-1, e1])
        data.append(['R-A-ADMM', k2-1, e2])
        data.append(['R-HB-ADMM', k3-1, e3])
        #data.append(['CVX', 0, e4])
    
    df = pd.DataFrame(data=data, columns=['algorithm', 'iteration', 'error'])
    pickle.dump(df, open('hist.pickle', 'wb'))

def plot_hist():
    df = pickle.load(open('hist.pickle', 'rb'))
    fig = plt.figure()
    fig, axes = plt.subplots(ncols=2,nrows=1,sharey=True)
    ax = axes.flat
    #e1 = df[df['algorithm']=='R-ADMM']['error'].to_numpy()
    #i1 = df[df['algorithm']=='R-ADMM']['iteration'].to_numpy()
    #e2 = df[df['algorithm']=='R-A-ADMM']['error'].to_numpy()
    #i2 = df[df['algorithm']=='R-A-ADMM']['iteration'].to_numpy()
    #e3 = df[df['algorithm']=='R-HB-ADMM']['error'].to_numpy()
    #i3 = df[df['algorithm']=='R-HB-ADMM']['iteration'].to_numpy()
    g = sns.histplot(df, x="error", y="algorithm", hue="algorithm", legend=False,
                    ax=ax[0])
    sns.histplot(df, x="iteration", y="algorithm", hue="algorithm", 
                 legend=False, ax=ax[1])
    for item in g.get_yticklabels():
        item.set_rotation(90)
    ax[0].set_ylabel('')
    plt.subplots_adjust(wspace=0.05,hspace=0.0)
    fig.savefig('l1trend_hist_error.pdf', bbox_inches='tight')


if __name__ == '__main__':

    n = 1000
    p = 0.99
    sigma = 20
    b = 0.5
    lamb = 15000
    rho = 1
    gamma = -np.log(0.99)
    #print(gamma)
    #mi = 1000
    mi = 5000
    tol = 1e-8
    
    y, x_true, F = l1trend.gen_data(n, p, sigma, b)
    
    #time_series(y, F, x_true, lamb,
    #            rho=rho, alpha=1, gamma=gamma, mi=mi, tol=tol)

    alpha_convergence2(y, F, x_true, lamb, rho=rho, gamma=gamma, 
                      mi=mi, amin=0.6,  amax=1.4, tol=tol)
    
    #many_trials(n, p, sigma, b, lamb, 80,
    #            rho=rho, gamma=gamma, mi=5000, tol=1e-5)
    #plot_hist()

