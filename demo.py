import pymc as pm
import pytensor.tensor as pt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mu_0 = np.random.default_rng().uniform(-50, 50)
sgm_0 = np.int_(1)
m = np.int_(0)
tau = np.int_(1)
#sgm_0 = np.random.default_rng().uniform(0.001, 10)
n = np.int_(256)
M = 1024

with (pm.Model() as model):
    N = pm.Normal.dist(mu=mu_0, sigma=sgm_0)
    x = pm.draw(N, draws=n)

    # Analytical
    T_n = 0.5 * np.log(2*np.pi * (n + 2)/(n + 1)) + (n + 1)/(2*(n + 2)) * (1/n) * np.sum((x - sum(x)/(n + 1))**2)

    # Monte Carlo approximation
    mu_pos = sum(x)/(n + 1)
    sgm_pos = 1/(n + 1)
    N_pos = pm.Normal.dist(mu=mu_pos, sigma=sgm_pos)

    mu = pt.vector("mu")
    N_i = pm.Normal.dist(mu, sgm_0)
    value = pt.scalar("value")
    logp_N = pm.logp(N_i, value)

    T_mc = np.double(0.0)
    for x_i in x:
        theta = pm.draw(N_pos, draws=M)
        p_x_i = np.exp(logp_N.eval({value: x_i, mu: theta}))
        r_i = np.mean(p_x_i)
        T_mc = T_mc + (-np.log(r_i))
    T_mc = T_mc/n

    # PyMC function
    

    #
    mu_x = np.mean(x)
    sgm_x = np.std(x)
    sns.histplot(x)
    plt.title("n = " + str(x.shape[0]) +
              ", mu = " + np.array2string(mu_x) +
              ", sgm = " + np.array2string(sgm_x))
    plt.show()

print("Hey\n")
