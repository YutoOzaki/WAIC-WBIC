import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable
import numpy as np
import scipy as sp
from typing import Optional, Tuple

def logp(value: TensorVariable, B: TensorVariable, kappa: TensorVariable) -> TensorVariable:
    p = B.type.shape[0]
    return (kappa/2)*pm.math.logdet(B) - (kappa+p+1)/2*pm.math.logdet(value) - 0.5*pt.trace(pt.matmul(B, pt.linalg.inv(value)))

def random(
        B: np.ndarray | float,
        kappa: np.ndarray | float,
        rng: Optional[np.random.Generator]=None,
        size: Optional[Tuple[int]]=None
) -> np.ndarray | float:
    return sp.stats.invwishart(df=kappa, scale=B).rvs(size=size)

def main(y):
    C = y.shape[1]

    with pm.Model():
        mu = pm.MvNormal('mu', mu=np.zeros(C), cov=10*np.eye(C))
        Sigma = pm.CustomDist('Sigma', np.eye(C), C+1,
                              logp=logp, random=random, shape=[C,C])
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

    print('Hey')
    print('Done')