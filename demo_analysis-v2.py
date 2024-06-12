import pymc as pm
import pytensor.tensor as pt
import numpy as np
import scipy as sp
import datetime
import arviz as az
import matplotlib.pyplot as plt

# Setup
nu_s = 2
A_s = 10
nu_σ = 2
A_σ = 10
nu_λ = 2
A_λ = 10
al = 0.25
K = 10
temperature = np.linspace(0, 1, K + 1)**(1/al)

def testdata(d):
    rng = np.random.default_rng()
    #C = rng.integers(low=10, high=40)
    C = 5
    D = sp.stats.invwishart(df=C+1, scale=2*np.eye(C)).rvs(size=1)
    u = np.expand_dims(rng.multivariate_normal(mean=np.zeros(C), cov=D), axis=1)
    N = rng.integers(low=128, high=256)
    c = np.concatenate([np.arange(0, C), rng.integers(low=0, high=C, size=N-C)], axis=0)
    Z = np.zeros(shape=(N, C))
    for i in range(0, N):
        Z[i, c[i]] = 1
    be = rng.uniform(low=-5, high=5, size=(d, 1))
    sgm = rng.uniform(low=0.001, high=2)
    X = np.concatenate([np.ones(shape=(N, 1)), rng.uniform(low=-10, high=10, size=(N, d-1))], axis=1)
    eps = rng.normal(loc=0, scale=sgm, size=(N, 1))
    y = X.dot(be) + 0*Z.dot(u) + eps
    return y, X, Z, c, u, be, sgm, D

def posteriorinference(y, X, Z, c, D, numdraw, numchain, numtune):
    d = X.shape[1]
    x_1 = X[:, 1]
    x_2 = X[:, 2]
    C = Z.shape[1]
    A_λ_rep = np.repeat(A_λ, C)

    with pm.Model() as model:
        Σ = 10 * np.eye(d)

        #λ_list = pm.math.stack([pm.HalfStudentT("λ_{}".format(i), nu=nu_λ, sigma=A_λ_rep[i]) for i in np.arange(C)])
        #R = pm.LKJCorr("R", n=C, eta=1, return_matrix=True)
        #λ_diag = pm.Deterministic("λ_diag", pt.diag(λ_list))
        #D = pm.Deterministic("D", pt.nlinalg.matrix_dot(λ_diag, R, λ_diag))
        #u = pt.shape_padaxis(pm.MvNormal("u", mu=np.zeros(C), cov=D), axis=1)

        # simplified
        s = pm.HalfStudentT("s", nu=nu_s, sigma=A_s)
        S = pm.Deterministic("S", pt.mul(s, np.eye(d)))

        m = pm.MvNormal("m", mu=np.zeros(d), cov=S)
        β = pt.shape_padaxis(pm.MvNormal("β", mu=m, cov=Σ), axis=1)
        μ = pm.Deterministic("μ", β[0] + β[1]*x_1 + β[2]*x_2)
        σ = pm.HalfStudentT("σ", nu=nu_σ, sigma=A_σ)
        pm.Normal("y", mu=μ, sigma=σ, observed=y)

        idata = pm.sample(draws=numdraw, chains=numchain, tune=numtune)

        return idata


if __name__ == "__main__":
    # generate data
    d = 3
    y, X, Z, c, u, be, sgm, D = testdata(d)

    # Setup - sampler
    numdraw = 1024
    numchain = 4
    numtune = 1000
    idata = posteriorinference(y, X, Z, c, D, numdraw, numchain, numtune)

    #az.plot_trace(idata, filter_vars="like", var_names=["β", "σ", "u", "D"])
    #plt.hist(y)
    print("hey!")
    print("{} Completed".format(datetime.datetime.now()))