# Use numpy
import numpy as np
import scipy as sp

# Setup
rng = np.random.default_rng()
N = rng.integers(low=2, high=10)
d = rng.integers(low=2, high=8)
M = 4096

y = rng.uniform(low=-2, high=2, size=(N, 1))
X = rng.uniform(low=-2, high=2, size=(N, d))
be = rng.uniform(low=-2, high=2, size=(d, 1))
V = rng.uniform(low=-3, high=3, size=(N, N))
V = np.matmul(V, V.T)

# test target
def mu(X, be):
    return X.dot(be)


def logmvnormal(y, mu, V):
    N = y.size
    v = y - mu
    l = -0.5 * v.T.dot(np.linalg.inv(V)).dot(v) \
        - 0.5 * np.log(np.linalg.det(V)) - 0.5 * N * np.log(2 * np.pi)
    return l


def dldbe(X, y, mu, V):
    return X.T.dot(np.linalg.inv(V)).dot(y - mu)


def sensimat(X, y, mu, V):
    S = np.linalg.inv(V)
    v = y - mu
    Q = v.dot(v.T).dot(S.T)
    return (X.T).dot(S).dot(Q).dot(X)


# Test likelihood function
l = logmvnormal(y, mu(X, be), V)
l_test = sp.stats.multivariate_normal.logpdf(np.squeeze(y), mean=np.squeeze(mu(X, be)), cov=V)
print("--log likelihood of multivariate normal--")
print(np.c_[l.flatten(), l_test.flatten(), (np.abs(l - l_test)/l).flatten()])

# Test gradient
s = dldbe(X, y, mu(X, be), V)
s_test = np.zeros((d, 1))
for i in range(0, d):
    eps = 1e-6
    be_d = be.copy()
    be_d[i] = be[i] - eps
    l_b = logmvnormal(y, mu(X, be_d), V)
    be_d[i] = be[i] + eps
    l_f = logmvnormal(y, mu(X, be_d), V)
    s_test[i] = (l_f - l_b)/(2*eps)
print("--gradient w.r.t. β--")
print(np.c_[s.flatten(), s_test.flatten(), (np.abs(s - s_test)/s).flatten()])

# Test sensitivity matrix
A = sensimat(X, y, mu(X, be), V)
A_test = np.matmul(s_test, s_test.T)
print("--sensitivity matrix--")
print(np.c_[A.flatten(), A_test.flatten(), (np.abs(A - A_test)/A).flatten()])

# Test convergence
Y = rng.multivariate_normal(mean=np.squeeze(mu(X, be)), cov=V, size=M).T
A_M = 0
for m in range(0, M):
    A_M = A_M + sensimat(X, Y[:, m].reshape((N, 1)), mu(X, be), V)
A_M = A_M/M
B = -X.T.dot(np.linalg.inv(V)).dot(X)
print("--Second Bartlett identity--")
print(np.c_[A_M.flatten(), B.flatten(), (np.abs(A_M - -B)/A_M).flatten()])

print("Completed")