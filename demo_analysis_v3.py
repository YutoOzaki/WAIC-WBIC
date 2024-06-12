import pymc as pm
import pytensor.tensor as pt
import numpy as np
import scipy as sp
import datetime
import arviz as az
import matplotlib.pyplot as plt

# Setup
nu_λ = 2
A_λ = 10
nu_s = 2
A_s = 10
nu_σ = 2
A_σ = 10
al = 0.25
K = 10
temperature = np.linspace(0, 1, K + 1)**(1/al)


def testdata(d):
    rng = np.random.default_rng()
    #N = rng.integers(low=128, high=256)
    #C = rng.integers(low=10, high=40)
    N = 300
    C = 8
    D = sp.stats.invwishart(df=C+1, scale=8*np.eye(C)).rvs(size=1)
    λ = np.sqrt(np.diagonal(D))
    R = np.linalg.inv(np.diag(λ)).dot(D).dot(np.linalg.inv(np.diag(λ)))
    u = np.expand_dims(rng.multivariate_normal(mean=np.zeros(C), cov=D), axis=1)
    c = np.concatenate([np.repeat(np.arange(0, C), 2), rng.integers(low=0, high=C, size=N-2*C)], axis=0)
    Z = np.zeros(shape=(N, C))
    for i in range(0, N):
        Z[i, c[i]] = 1
    be = rng.uniform(low=-8, high=8, size=(d, 1))
    sgm = rng.uniform(low=0.001, high=2)
    X = np.concatenate([np.ones(shape=(N, 1)), rng.uniform(low=-10, high=10, size=(N, d-1))], axis=1)
    eps = rng.normal(loc=0, scale=sgm, size=(N, 1))
    y = X.dot(be) + Z.dot(u) + eps
    return {'y':y, 'X':X, 'Z':Z, 'c':c, 'u':u, 'be':be, 'sgm':sgm, 'D':D, 'λ':λ, 'R':R}


def posteriorinference(prm, testprm, numdraw, numchain, numtune, tarate):
    y = prm['y']
    X = prm['X']
    Z = prm['Z']

    d = X.shape[1]
    C = Z.shape[1]
    A_λ_rep = np.repeat(A_λ, C)

    with pm.Model() as model:
        if testprm == 'u':
            D = prm['D']
            u = pt.shape_padaxis(pm.MvNormal("u", mu=np.zeros(C), cov=D), axis=1)
        else:
            if testprm == 'λ':
                λ = prm['λ']
            else:
                #λ_list = pm.math.stack([pm.HalfStudentT("λ_{}".format(i), nu=nu_λ, sigma=A_λ_rep[i]) for i in np.arange(C)])
                #λ_diag = pm.Deterministic("λ_diag", pt.diag(λ_list))
                λ = pm.HalfStudentT("λ", nu=nu_λ, sigma=A_λ_rep)
                #a = pm.InverseGamma("a", alpha=0.5, beta=1/(A_λ_rep**2))
                #λ = pm.InverseGamma("λ", alpha=nu_λ/2, beta=nu_λ/a)
                #λ = pm.LogNormal("λ", mu=np.zeros(C), sigma=A_λ_rep)
            if testprm == 'R':
                R = prm['R']
            else:
                R = pm.LKJCorr("R", n=C, eta=1, return_matrix=True)
            #D = pm.Deterministic("D", pt.nlinalg.matrix_dot(λ_diag, R, λ_diag))
            z = pm.MvNormal("z", mu=np.zeros(C), cov=R)
            u = pm.Deterministic("u", pt.math.matmul(pt.diag(λ), pt.shape_padaxis(z, axis=1)))

        if testprm == 'u' or testprm == 'D' or testprm == 'R' or testprm == 'λ':
            β = prm['be']
            σ = prm['sgm']
        else:
            # simplified
            s = pm.HalfStudentT("s", nu=nu_s, sigma=A_s)
            S = pm.Deterministic("S", pt.mul(s, np.eye(d)))
            m = pm.MvNormal("m", mu=np.zeros(d), cov=S)
            #m = np.zeros(d)
            Σ = 10 * np.eye(d)
            β = pt.shape_padaxis(pm.MvNormal("β", mu=m, cov=Σ), axis=1)
            σ = pm.HalfStudentT("σ", nu=nu_σ, sigma=A_σ)

        μ = pm.Deterministic("μ", pm.math.matmul(X, β) + pm.math.matmul(Z, u))
        # μ = pm.Deterministic("μ", pm.math.matmul(X, β))
        pm.Normal("y", mu=μ, sigma=σ, observed=y)
        #pm.MvNormal("y", mu=μ, cov=pt.mul(σ, np.eye(y.shape[0])), observed=y)

        idata = pm.sample(draws=numdraw, chains=numchain, tune=numtune, target_accept=tarate)
        #idata = pm.fit(n=10000, method='advi', callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])

        return idata


if __name__ == "__main__":
    # generate data
    d = 3
    prm = testdata(d)
    testprm = 'λ'

    # Setup - sampler
    numdraw = 300
    numchain = 4
    numtune = 300
    tarate = 0.9
    idata = posteriorinference(prm, testprm, numdraw, numchain, numtune, tarate)

    az.plot_trace(idata, filter_vars="regex", var_names=["β", "σ", "u", "λ", "R"])
    #print(np.nanmean(idata.posterior["R"].to_numpy(), axis=(0, 1)))
    #print(np.nanmean(idata.posterior["λ"].to_numpy(), axis=(0, 1)))
    #plt.hist(y)
    print("hey!")
    print("{} Completed".format(datetime.datetime.now()))

