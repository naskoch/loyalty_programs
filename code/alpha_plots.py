import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def comp_rev(k, alpha, v, lam, p, beta, delta):
    return (1.-alpha*v)*(p*k*lam/(k-(1.-lam)*delta)+(1.-p)*lam)

def compute_delta(v, R, beta):
	return max(0.,np.log(v/(R*(1.-beta)))/np.log(beta))

def sample_unif(b, n):
    lambdas = np.random.uniform(0.,b,n)
    return lambdas

def sample_logit_norm(n):
    lambdas = np.random.randn(n)
    lambdas = [math.exp(k)/(1+math.exp(k)) for k in lambdas]
    return lambdas

def plot_alpha_unif(b, p, v, beta, n):
    alphas = np.linspace(0,math.e, 250)
    t = len(alphas)
    rev_avgs = np.zeros((t,1))
    for j in range(t):
        lambdas = sample_unif(b, n)
        alpha = alphas[j]
        k = math.e/(alpha*(1.-beta))
        delta = compute_delta(v, alpha*k*v, beta)
        rev_avgs[j] = sum([comp_rev(k, alpha, v, lam, p, beta, delta) for lam in lambdas])/n
    plt.plot(alphas, rev_avgs)
    plt.show()

def plot_alpha_logit(p, v, beta, n):
    alphas = np.linspace(0,math.e, 250)
    t = len(alphas)
    rev_avgs = np.zeros((t,1))
    for j in range(t):
        lambdas = sample_logit_norm(n)
        alpha = alphas[j]
        k = math.e/(alpha*(1.-beta))
        delta = compute_delta(v, alpha*k*v, beta)
        rev_avgs[j] = sum([comp_rev(k, alpha, v, lam, p, beta, delta) for lam in lambdas])/n
    plt.plot(alphas, rev_avgs)
    plt.show()

b = 1.
p = 0.9
v = 0.18
beta = 0.9
n = 10000

plot_alpha_unif(b, p, v, beta, n)