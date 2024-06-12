import pymc as pm
import pytensor.tensor as pt
import numpy as np
import scipy as sp
import arviz as az
from datetime import datetime

def main(y, Z, prm):
    C = Z.shape[1]
    #R = prm['sgm']*np.eye(y.shape[0])
    sgm = prm['sgm']

    with pm.Model():
        mu = pm.MvNormal('mu', mu=np.zeros(C), cov=10*np.eye(C))
        λ = pm.HalfStudentT.dist(nu=2, sigma=np.repeat(10, C))
        L, corr, sigmas = pm.LKJCholeskyCov('L', eta=1, n=C, sd_dist=λ)
        D = pm.Deterministic('D', pm.math.matmul(L, L.T))
        u = pm.MvNormal('u', mu=mu, cov=D)

        μ = pm.Deterministic('μ', pm.math.matmul(Z, u))
        #pm.MvNormal('y', mu=μ, cov=R, observed=y)
        pm.Normal('y', mu=μ, sigma=sgm, observed=y)

        idata = pm.sample(draws=256, chains=4, tune=200)
        return idata

if __name__ == "__main__":
    N = 512
    C = 32
    Z = np.random.multinomial(1, [1/C]*C, size=N)
    mu_0 = np.random.uniform(low=-12, high=12, size=C)
    Sigma_0 = sp.stats.invwishart(df=C+1, scale=5*np.eye(C)).rvs()
    u = sp.stats.multivariate_normal(mean=mu_0, cov=Sigma_0).rvs()
    sgm_0 = np.random.uniform(low=0.001, high=3)
    eps = np.random.normal(loc=0, scale=sgm_0, size=N)
    y = Z.dot(u) + eps

    print(datetime.now)
    idata = main(y, Z, {'sgm': sgm_0})
    az.plot_trace(idata, filter_vars="regex", var_names=['u', 'L_stds'])
    print(np.stack((u, np.nanmean(idata.posterior["u"].to_numpy(), axis=(0, 1)))).T)
    print(np.stack((
        np.sqrt(np.diagonal(Sigma_0)),
        np.nanmedian(idata.posterior["L_stds"].to_numpy(), axis=(0, 1)))
    ).T)

    print('Completed')