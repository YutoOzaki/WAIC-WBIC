import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable
from typing import Optional, Tuple
import numpy as np
import scipy as sp
import arviz as az
import matplotlib.pyplot as plt
from datetime import datetime

def logp_nonlocal(value:TensorVariable, mu: TensorVariable, Sigma: TensorVariable) -> TensorVariable:
    p = mu.type.shape[0]
    z = value - mu
    return (2*pm.math.log(pm.math.abs(value[1])) - pm.math.log(Sigma[1,1])
            -(p/2)+pm.math.log(2*np.pi) - 0.5*pm.math.logdet(Sigma) -
            0.5*pt.linalg.matrix_dot(z.T, pt.linalg.inv(Sigma), z))

def random_nonlocal(mu: np.ndarray | float, Sigma: np.ndarray | float,
                    rng: Optional[np.random.Generator] = None,
                    size : Optional[Tuple[int]]=None) -> np.ndarray | float:
    print("start rejection sampling")

    p = mu.shape[0]
    g = sp.stats.multivariate_normal(mean=mu, cov=Sigma, seed=rng)
    f = sp.stats.multivariate_t(df=p+1, shape=Sigma, seed=rng)

    i=1
    Y = f.rvs(size)
    u = np.random.uniform(0, 1)
    r = (Y[1]**2)/Sigma[1,1]*g.pdf(Y) / (4*d*f.pdf(Y))
    while np.all(u >= r):
        Y = f.rvs()
        u = np.random.uniform(0, 1)
        r = (Y[1]**2)/Sigma[1, 1]*g.pdf(Y) / (4*d*f.pdf(Y))
        i=i+1

    print("Rejection sampling trial i={}".format(i))

    return Y

def momdist_nonlocal(rv: TensorVariable, size: TensorVariable, mu: TensorVariable, Sigma: TensorVariable) -> TensorVariable:
  return mu + np.sqrt(2)*pt.sqrt(pt.diagonal(Sigma))

def dist_nonlocal(mu: TensorVariable, Sigma: TensorVariable, size: TensorVariable) -> TensorVariable:
    be = pm.MvNormal.dist(mu=mu, cov=Sigma, size=size)
    return (be[1]**2)*be

def main(y, Z, c, X, prm):
    N, C = Z.shape
    d = X.shape[1]
    iX = np.linalg.pinv(X)
    #R = prm['sgm']*np.eye(y.shape[0])

    with pm.Model():
        λ = pm.HalfStudentT.dist(nu=2, sigma=np.repeat(1000, C))
        L, corr, sigmas = pm.LKJCholeskyCov('L', eta=1, n=C, sd_dist=λ)
        D = pm.Deterministic('D', pm.math.matmul(L, L.T))
        u = pm.MvNormal('u', mu=np.zeros(C), cov=D)
        #z = pm.Normal('z', mu=0, sigma=1, size=C)
        #u = pm.Deterministic('u', pt.dot(L, z))
        s = pm.HalfStudentT('s', nu=2, sigma=1000)
        #Σ = pm.Deterministic('Σ', N*pt.linalg.matrix_dot(iX, pt.linalg.matrix_dot(Z,D,Z.T) + s*np.eye(N), iX.T))
        Σ = N * pt.linalg.matrix_dot(iX, pt.linalg.matrix_dot(Z, D, Z.T) + s * np.eye(N), iX.T)
        #β = pm.CustomDist('β', np.zeros(d), Σ, dist=dist_nonlocal, logp=logp_nonlocal)
        #β = pt.squeeze(pm.CustomDist('β', np.zeros(d), Σ, logp=logp_nonlocal, random=random_nonlocal, shape=(3,1)))
        β = pm.CustomDist('β', np.zeros(d), Σ, logp=logp_nonlocal, random=random_nonlocal, moment=momdist_nonlocal, signature='()->()')
        #β = pm.MvNormal('β', mu=np.zeros(d), cov=Σ)
        #β = pm.MvNormal('β', mu=np.zeros(d), cov=10*np.eye(d))
        # μ = pm.Deterministic('μ', pm.math.matmul(Z, u))
        # pm.MvNormal('y', mu=μ, cov=R, observed=y)
        #μ = pm.Deterministic('μ', pm.math.matmul(X, β) + u[c])
        μ = pm.math.matmul(X, β) + u[c]
        pm.Normal('y', mu=μ, sigma=s, observed=y)
        #pm.Normal('y', mu=μ, sigma=prm['sgm'], observed=y)
        #V = pm.Deterministic('V', pt.linalg.matrix_dot(Z, D, Z.T) + R)
        #pm.MvNormal('y', mu=np.zeros(N), cov=V, observed=y)

        idata = pm.sample(draws=256, chains=4, tune=200, target_accept=0.85)
        return idata

if __name__ == "__main__":
    N = 512
    C = 32
    d = 3
    Z = np.random.multinomial(1, [1/C]*C, size=N)
    c = np.squeeze(np.concatenate([np.where(Z[i, :]==1) for i in range(N)]))
    D_0 = sp.stats.invwishart(df=C+1, scale=5*np.eye(C)).rvs()
    u = sp.stats.multivariate_normal(mean=np.zeros(C), cov=D_0).rvs()
    sgm_0 = np.random.uniform(low=0.001, high=3)
    eps = np.random.normal(loc=0, scale=sgm_0, size=N)
    be = np.random.uniform(low=-3, high=3, size=d)
    X = np.stack([np.ones(N), np.tile((0,1), int(N/2)), np.random.normal(loc=0, scale=5, size=N)], axis=1)
    y = X.dot(be) + Z.dot(u) + eps

    print(datetime.now())
    idata = main(y, Z, c, X, {'sgm': sgm_0})
    az.plot_trace(idata, filter_vars="regex", var_names=['β', 's', 'u'])
    az.plot_energy(idata)
    az.summary(idata, round_to=2)
    print(np.stack((be, np.nanmean(idata.posterior["β"].to_numpy(), axis=(0, 1)))).T)
    print(np.stack((u, np.nanmean(idata.posterior["u"].to_numpy(), axis=(0, 1)))).T)
    print(np.stack((
        np.sqrt(np.diagonal(D_0)),
        np.nanmedian(idata.posterior["L_stds"].to_numpy(), axis=(0, 1)))
    ).T)
    plt.hist(idata.posterior["β"].to_numpy()[:,:,1].flatten(), bins=50)


    print('Completed')

'''
x = np.linspace(-8, 8, 2048)
M = 8192
test = np.zeros(M)
for i in range(M):
    s = np.random.uniform(0.001, 6)
    f = (x**2)/(s**2)*sp.stats.norm(scale=s).pdf(x)
    g = 4 * sp.stats.t(df=2, scale=s).pdf(x)
    test[i] = np.all(g >= f)
print(np.all(test))

plt.plot(x, f)
plt.plot(x, g)
y = np.sqrt(2)*s
plt.stem(y, (y**2)/(s**2)*sp.stats.norm(scale=s).pdf(y), markerfmt=" ")
plt.stem(-y, (y**2)/(s**2)*sp.stats.norm(scale=s).pdf(-y), markerfmt=" ")

M = 8192
test = np.zeros(M)
d = 3
q = sp.stats.invwishart(df=d+1, scale=np.eye(d))
for i in range(M):
    s = q.rvs()
    x = np.random.uniform(-10, 10, size=d)
    f = (x[1]**2)/s[1,1]*sp.stats.multivariate_normal(mean=np.zeros(d), cov=s).pdf(x)
    g = d*4 * sp.stats.multivariate_t(df=d+1, shape=s).pdf(x)
    test[i] = g >= f
print(np.all(test))

a = pm.CustomDist('a', np.zeros(d), np.eye(d), logp=logp_nonlocal, random=random_nonlocal)
b = pm.draw(a, draws=512)

c = np.zeros((d, 128))
for i in range(128):
    c[:, i] = random_nonlocal(np.zeros(d), np.eye(d))
import matplotlib.pyplot as plt
plt.hist(c[1, :])

a = pm.CustomDist('a', np.zeros(d), np.eye(d), logp=logp_nonlocal, random=random_nonlocal, shape=(3,))
b = pm.draw(a, draws=12)
'''