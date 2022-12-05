# Accelerated Extensions of ADMM

Code related to the paper

* ![G. França, D. P. Robinson, R. Vidal, "A nonsmooth dynamical systems perspective on accelerated extensions of ADMM"](https://arxiv.org/abs/1808.04048)

We propose two extensions of the (known) relaxed Alternating Direction Method of Multipliers (R-ADMM), called R-A-ADMM and R-HB-ADMM, which include relaxation and acceleration. These variants are faster than the original R-ADMM.

Each "directory" contains code for a type of problem.

As an example, one problem consists in finding piecewise linear trends in time series, as illustrated below: 

![](https://github.com/guisf/accelerated_ADMM/blob/main/figs/l1trending.png)

The convergence rate of the methods are illustrated below:

![](https://github.com/guisf/accelerated_ADMM/blob/main/figs/l1trend_alpha.png)

Another problem we consider is Robust Principal Component Analysis (RPCA), where two components are recoverred from
an observed matrix, i.e., a low rank matrix and a sparse matrix. This problem has many interesting applications.
The convergence rates of our methods are illustrated below:

![](https://github.com/guisf/accelerated_ADMM/blob/main/figs/rpca1.png)

