import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# For debug
import pytensor.tensor as pt
from pytensor import function
from pytensor.printing import Print

d = 4
m = np.zeros(d)
I = np.eye(d)

N = 512
rng = np.random.default_rng()
data = pd.DataFrame({"x1": np.ones(N),
                     "x2": rng.normal(2, 0.8, size=N),
                     "x3": rng.normal(-1, 2, size=N),
                     "x4": rng.normal(1, 0.6, size=N)})
ε = rng.normal(0, 0.2, size=N)
data = data.assign(y=1.26*data['x1'] - 0.24*data['x2'] + 2.19*data['x3'] - 0.75*data['x4'] + ε)


def run_model():
    with pm.Model() as LMM:
        X = pm.MutableData("X", data[['x1', 'x2', 'x3', 'x4']])
        σ_β = pm.HalfStudentT("σ_β", nu=2, sigma=10)
        D = pm.Deterministic("D", pm.math.dot(σ_β, I))
        β = pm.MvNormal("β", mu=m, cov=D)
        μ = pm.Deterministic("μ", pm.math.dot(X, β))
        σ = pm.HalfStudentT("σ", nu=2, sigma=10)
        pm.Normal("y", mu=μ, sigma=σ, observed=data.y)

    with LMM:
        idata = pm.sample_smc()

    return idata


if __name__ == '__main__':
    idata = run_model()
    print(np.nanmean(np.nanmean(idata.sample_stats["log_marginal_likelihood"].data)))
    az.plot_trace(idata, filter_vars="regex", var_names=["β"])
    print("Completed")
