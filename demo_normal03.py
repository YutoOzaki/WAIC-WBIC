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


def demo_marginallikelihood():
    with (pm.Model() as model):
        N = pm.Normal.dist(mu=mu, sigma=np.sqrt(sgmsq))
        x = pm.draw(N, draws=n)

        # Analytical
        logZ_n = np.log(np.sqrt(sgmsq)) - n*np.log(np.sqrt(2*np.pi*sgmsq)) - 0.5*np.log(n*sgmsq_0 + sgmsq) +\
              -sum(x**2)/(2*sgmsq) - mu_0**2/(2*sgmsq_0) +\
              (sgmsq_0*n**2*np.mean(x)**2/sgmsq + sgmsq*mu_0**2/sgmsq_0 + 2*n*np.mean(x)*mu_0)/(2*(n*sgmsq_0 + sgmsq))

        # Monte Carlo approximation
        N_pri = pm.Normal.dist(mu=mu_0, sigma=np.sqrt(sgmsq_0))
        mu_i = pt.scalar("mu_i")
        N_i = pm.Normal.dist(mu=mu_i, sigma=np.sqrt(sgmsq))
        sample = pt.vector("sample")
        logp_N = pm.logp(N_i, sample)

        theta = pm.draw(N_pri, draws=M)
        logZ_mc = np.double(0.0)
        for theta_i in theta:
            logZ_mc = logZ_mc + np.sum(logp_N.eval({sample: x, mu_i: theta_i}))
        logZ_mc = logZ_mc/M


        # PyMC function
        N_mu = pm.Normal("mu", mu=mu_0, sigma=np.sqrt(sgmsq_0))
        N_x = pm.Normal("obs", mu=N_mu, sigma=np.sqrt(sgmsq), observed=x)
        mu_post = pm.sample(draws=M)
        pm.compute_log_likelihood(mu_post)
        pooled_loo = az.loo(mu_post)
        T_pm = -pooled_loo.elpd_loo/n

    return x, logZ_n, logZ_mc, T_pm


def visualize(x):
    mu_x = np.mean(x)
    sgm_x = np.std(x)
    sns.histplot(x)
    plt.title("n = " + str(x.shape[0]) +
              ", mu = " + np.array2string(mu_x) +
              ", sgm = " + np.array2string(sgm_x))
    plt.show()


if __name__ == "__main__":
    x, T_n, T_mc, T_pm = demo_marginallikelihood()

    print("Empirical loss\n Analytical: " + np.array2string(T_n) +
          "\n Monte Carlo approximation: " + np.array2string(T_mc) +
          "\n PyMC function (elpd_loo): " + np.array2string(T_pm) + "\n")

    visualize(x)
