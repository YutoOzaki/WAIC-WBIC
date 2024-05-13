import pymc as pm
import pytensor.tensor as pt
import arviz as az
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n = np.int_(256)
mu = np.random.default_rng().uniform(-50, 50)
sgmsq = np.random.default_rng().uniform(0.001, 10)
mu_0 = np.random.default_rng().uniform(-50, 50)
sgmsq_0 = np.random.default_rng().uniform(0.001, 10)
M = 1024


def demo_empiricalloss():
    with (pm.Model() as model):
        N = pm.Normal.dist(mu=mu, sigma=np.sqrt(sgmsq))
        x = pm.draw(N, draws=n)

        # Analytical
        mu_pos = (sgmsq*mu_0 + n*sgmsq_0*np.mean(x))/(n*sgmsq_0 + sgmsq)
        sgmsq_pos = sgmsq*sgmsq_0/(n*sgmsq_0 + sgmsq)
        N_pred = pm.Normal.dist(mu=mu_pos, sigma=np.sqrt(sgmsq_pos + sgmsq))
        value = pt.vector("value")
        logp_N = pm.logp(N_pred, value)
        p = np.exp(logp_N.eval({value: x}))
        T_n = np.mean(-np.log(p))

        # Monte Carlo approximation
        N_pos = pm.Normal.dist(mu=mu_pos, sigma=np.sqrt(sgmsq_pos))
        mu_i = pt.vector("mu_i")
        N_i = pm.Normal.dist(mu=mu_i, sigma=np.sqrt(sgmsq))
        sample = pt.scalar("sample")
        logp_N = pm.logp(N_i, sample)

        T_mc = np.double(0.0)
        for x_i in x:
            theta = pm.draw(N_pos, draws=M)
            p_x_i = np.exp(logp_N.eval({sample: x_i, mu_i: theta}))
            r_i = np.mean(p_x_i)
            T_mc = T_mc + (-np.log(r_i))
        T_mc = T_mc/n

        # PyMC function
        N_mu = pm.Normal("mu", mu=mu_0, sigma=np.sqrt(sgmsq_0))
        N_x = pm.Normal("obs", mu=N_mu, sigma=np.sqrt(sgmsq), observed=x)
        mu_post = pm.sample(draws=M)
        pm.compute_log_likelihood(mu_post)
        pooled_loo = az.loo(mu_post)
        T_pm = -pooled_loo.elpd_loo/n

    return x, T_n, T_mc, T_pm


def visualize(x):
    mu_x = np.mean(x)
    sgm_x = np.std(x)
    sns.histplot(x)
    plt.title("n = " + str(x.shape[0]) +
              ", mu = " + np.array2string(mu_x) +
              ", sgm = " + np.array2string(sgm_x))
    plt.show()


if __name__ == "__main__":
    x, T_n, T_mc, T_pm = demo_empiricalloss()

    print("Empirical loss\n Analytical: " + np.array2string(T_n) +
          "\n Monte Carlo approximation: " + np.array2string(T_mc) +
          "\n PyMC function (elpd_loo): " + np.array2string(T_pm) + "\n")

    visualize(x)
