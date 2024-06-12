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


def testdata():
    rng = np.random.default_rng()
    #N = rng.integers(low=128, high=256)
    #C = rng.integers(low=10, high=40)

    N = 400
    X = np.concatenate(
        [np.ones(shape=(N,1)),
         np.expand_dims(np.tile([0,1], int(N/2)), axis=1),
         rng.normal(loc=55, scale=5, size=(N,1))], axis=1)

    C = 25
    c = np.repeat(np.arange(0, C), N/C)
    Z = np.zeros(shape=(N, C))
    for i in range(0, N):
        Z[i, c[i]] = 1

    D = sp.stats.invwishart(df=C+1, scale=8*np.eye(C)).rvs(size=1)
    λ = np.sqrt(np.diagonal(D))
    R = np.linalg.inv(np.diag(λ)).dot(D).dot(np.linalg.inv(np.diag(λ)))
    u = np.expand_dims(rng.multivariate_normal(mean=np.zeros(C), cov=D), axis=1)

    be = np.expand_dims([rng.uniform(low=0, high=10), rng.uniform(low=5, high=20), rng.uniform(low=0.8, high=1.1)], axis=1)
    sgm = rng.uniform(low=0.001, high=3)
    eps = rng.normal(loc=0, scale=sgm, size=(N, 1))

    y = X.dot(be) + Z.dot(u) + eps
    return {'y':y, 'X':X, 'Z':Z, 'c':c, 'u':u, 'be':be, 'sgm':sgm, 'D':D, 'λ':λ, 'R':R}


def posteriorinference(prm, testprm, numdraw, numchain, numtune, tarate):
    y = prm['y']
    X = prm['X']
    Z = prm['Z']

    d = X.shape[1]
    C = Z.shape[1]
    N = X.shape[0]
    A_λ_rep = np.repeat(A_λ, C)

    with pm.Model() as model:
        if testprm == 'be':
            D = prm['D']
            u = pt.shape_padaxis(pm.MvNormal('u', mu=np.zeros(C), cov=D), axis=1)
        else:
            λ = pm.HalfStudentT.dist(nu=nu_λ, sigma=A_λ_rep)
            L, corr, sigmas = pm.LKJCholeskyCov('L', eta=1, n=C, sd_dist=λ)
            z = pm.Normal('z', mu=0, sigma=1, size=C)
            u = pm.Deterministic('u', pt.dot(L, z))
            D = pm.Deterministic('D', pt.dot(L, L.T))

        if testprm == 'u':
            β = prm['be']
            σ = prm['sgm']
        else:
            σ = pm.HalfStudentT("σ", nu=nu_σ, sigma=A_σ)

            # simplified
            s = pm.HalfStudentT("s", nu=nu_s, sigma=A_s)
            S = pm.Deterministic("S", s*pt.eye(d))
            m = pm.MvNormal("m", mu=np.zeros(d), cov=S)
            #m = np.zeros(d)

            R = pm.Deterministic('R', σ*pt.eye(N))
            V = pm.Deterministic('V', pt.nlinalg.matrix_dot(Z, D, Z.T) + R)
            iV = pm.Deterministic('iV', pt.nlinalg.inv(V))
            A = pm.Deterministic('A', pt.nlinalg.matrix_dot(-X.T, iV, X))
            #q = pm.Deterministic('q', y - pm.math.matmul(X, pt.shape_padaxis(m, axis=1)))
            #Q = pm.Deterministic('Q', pt.nlinalg.matrix_dot(q, q.T, iV))
            #B = pm.Deterministic('B', pt.nlinalg.matrix_dot(X.T, iV, Q, X))
            #Λ = pm.Deterministic('Λ', (1/N)*pt.nlinalg.matrix_dot(A, pt.nlinalg.inv(B), A))
            Λ = pm.Deterministic('Λ', (1 / N) * A)
            #Σ = 10 * pt.eye(d)

            β = pt.shape_padaxis(pm.MvNormal("β", mu=m, tau=Λ), axis=1)
            #β = pt.shape_padaxis(pm.MvNormal("β", mu=m, cov=Σ), axis=1)

        μ = pm.Deterministic('μ', pm.math.matmul(X, β) + pm.math.matmul(Z, u))
        pm.Normal('y', mu=μ, sigma=σ, observed=y)

        idata = pm.sample(draws=numdraw, chains=numchain, tune=numtune, target_accept=tarate)
        #idata = pm.fit(n=10000, method='svgd', callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])

        return idata


if __name__ == "__main__":
    # generate data
    prm = testdata()
    testprm = 'be'

    # Setup - sampler
    numdraw = 300
    numchain = 4
    numtune = 300
    tarate = 0.95
    idata = posteriorinference(prm, testprm, numdraw, numchain, numtune, tarate)

    az.plot_trace(idata, filter_vars="regex", var_names=['β','σ','u','L_stds'])
    #az.plot_trace(idata.sample(1024), filter_vars="regex", var_names=['β', 'σ', 'u', 'z'])
    #print(np.nanmean(idata.posterior["L_corr"].to_numpy(), axis=(0, 1)))
    #print(np.nanmean(idata.posterior["L_stds"].to_numpy(), axis=(0, 1)))
    #plt.hist(prm['y'], density=True, bins=32)
    #plt.hist(idata.posterior["μ"].to_numpy().flatten(), density=True, bins=32)
    print("hey!")
    print("{} Completed".format(datetime.datetime.now()))

