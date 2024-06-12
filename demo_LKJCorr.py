import pymc as pm
import pytensor.tensor as pt
import numpy as np
import scipy as sp
import arviz as az

def main(y):
    C = y.shape[1]

    with pm.Model():
        mu = pm.MvNormal('mu', mu=np.zeros(C), cov=10*np.eye(C))
        λ = pm.HalfStudentT.dist(nu=2, sigma=np.repeat(10, C))
        L, corr, sigmas = pm.LKJCholeskyCov('L', eta=1, n=C, sd_dist=λ)
        Sigma = pm.Deterministic('Sigma', pm.math.matmul(L, L.T))
        pm.MvNormal('y', mu=mu, cov=Sigma, observed=y)
        idata = pm.sample(draws=1024, chains=4, tune=1000)
        return idata

if __name__ == "__main__":
    C = 5
    N = 128
    mu_0 = np.random.uniform(low=-8, high=8, size=C)
    Sigma_0 = sp.stats.invwishart(df=C+1, scale=5*np.eye(C)).rvs()
    y = sp.stats.multivariate_normal(mean=mu_0, cov=Sigma_0).rvs(size=(N, 1))

    idata = main(y)
    az.plot_trace(idata, filter_vars="regex", var_names=['mu', 'L_stds'])
    print(mu_0)
    print(np.sqrt(np.diagonal(Sigma_0)))

    print('Completed')