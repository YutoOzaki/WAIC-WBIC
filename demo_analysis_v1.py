import pymc as pm
import pytensor
import pytensor.tensor as pt
import numpy as np
import datetime

# Setup
d = 3
A_S = 10*np.ones(d)
C = 40
A = 10*np.ones(C)
nu_0 = 2
al = 0.25
K = 10
temperature = np.linspace(0, 1, K + 1)**(1/al)

# Test data
rng = np.random.default_rng()
G = rng.uniform(low=-3, high=3, size=(d, d))
G = G.dot(G.T)
sgm_0 = 0.875
N = rng.integers(low=64, high=256)
be = np.expand_dims(rng.multivariate_normal(mean=np.arange(1, d+1), cov=G), axis=1)
X = rng.uniform(low=-10, high=10, size=(N, d))
eps = rng.normal(loc=0, scale=sgm_0, size=(N, 1))
y = X.dot(be) + eps


def validation():
    d = 3
    A_S = 10 * np.ones(d)
    with pm.Model():
        p = pm.InverseGamma("a", alpha=0.5, beta=1/A_S**2, shape=d)
        pytensor.dprint(p)


def logWishdist(x, a, nu):
    p = a.shape[0]
    B = pt.diag(1/a)
    lnG = p*(p-1)/4*pm.math.log(np.pi) + pm.math.sum([pt.math.gammaln((nu+1-i)/2) for i in range(1, d+1)])
    return nu/2*pm.math.logdet(B) - (nu+p+1)/2*pm.math.logdet(x) \
            - 0.5*pt.trace(pm.math.matmul(B, pm.math.matrix_inverse(x))) \
            - nu*p/2*pm.math.log(2) - lnG


def posteriorinference(temperature, numdraw, numchain, numtune):
    with pm.Model() as model:
        p_a = pm.InverseGamma("a", alpha=0.5, beta=1/A_S**2, shape=d)
        #p_a = []
        #for i in range(0, d):
        #    p_a.append(pm.InverseGamma("a_{}".format(i), alpha=0.5, beta=1/A_S[i]**2))
        p_S = pm.CustomDist("S", 2*nu_0*p_a, nu_0+d-1, logp=logWishdist, shape=(d, d))
        p_m = pm.MvNormal("m", mu=np.zeros(d), cov=p_S)
        p_be = pm.MvNormal("be", mu=p_m, cov=G)
        x = pm.MutableData("x", X)
        μ = pm.Deterministic("μ", pm.math.matmul(x, p_be))
        pm.MvNormal("y", mu=μ, cov=np.eye(N)*sgm_0, observed=y)

        print("hey")
        idata = pm.sample(draws=numdraw, chains=numchain, tune=numtune)

        return idata


if __name__ == "__main__":
    # Debugging
    validation()

    # Setup - sampler
    numdraw = 1024
    numchain = 4
    numtune = 1000

    # Posterior inference
    for k in range(0, K):
        posteriorinference(temperature[k], numdraw, numchain, numtune)

    print("{} Completed".format(datetime.datetime.now()))

# small test #1
# f.eval({x_pt: 3.22, mu_pt: 0.983, sgm_pt: 1.1112, be_pt: 0.1})
# 0.1*norm.logpdf(3.22, 0.983, 1.1112)
# np.log(norm.pdf(3.22, 0.983, 1.1112)**0.1)

# small test #2
# idata = main(0.0)
# mu_pos = idata.posterior["mu"].to_numpy().flatten()
# sgm_pos = idata.posterior["sigma"].to_numpy().flatten()
# mu_pri = rng.normal(loc=mu_0, scale=sgm_0, size=M*J)
# sgm_pri = np.abs(rng.standard_t(nu_0, size=M*J))
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 2)
# axes[0].hist(mu_pos, bins=50)
# axes[1].hist(mu_pri, bins=50)
# fig, axes = plt.subplots(1, 2)
# axes[0].hist(sgm_pos, bins=50)
# axes[1].hist(sgm_pri, bins=50)
# from scipy.special import gamma
# print(2*np.sqrt(nu_0/np.pi)*gamma((nu_0+1)/2)/(gamma(nu_0/2)*(nu_0 - 1)))

# small test #3
# al = 2.23; be = 3.01
# from scipy.stats import invgamma
# x = invgamma.rvs(a=al, scale=be, size=4096)
# print([np.mean(x), be/(al - 1), np.var(x), be**2/((al - 1)**2*(al - 2))])
# v = np.var(invgamma.rvs(a=al, scale=be, size=(4096, 128)), axis=0)
# import matplotlib.pyplot as plt
# plt.hist(v)
# x = 1/rng.gamma(shape=al, scale=1/be, size=4096)
# print([np.mean(x), be/(al - 1), np.var(x), be**2/((al - 1)**2*(al - 2))])
# plt.hist(x)

# small test 4
# from scipy.special import gamma
# A = 4.321; nu = 4;
# x = np.sqrt([1/rng.gamma(shape=nu/2, scale=(nu/a)**(-1)) for a in 1/rng.gamma(shape=0.5, scale=(1/A**2)**(-1), size=4096)])
# y = A*np.abs(rng.standard_t(nu, 4096))
# print((np.mean(x), np.mean(y), 2*A*np.sqrt(nu/np.pi)*gamma((nu+1)/2)/(gamma(nu/2)*(nu-1))))
# print((np.var(x), np.var(y), A**2*(nu/(nu-2) - 4*nu/(np.pi*(nu-1)**2) * (gamma((nu+1)/2)/gamma(nu/2))**2)))