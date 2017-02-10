import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def comp_rev(k, alpha, v, lam, p, beta):
    return (1.-alpha*v)*(p*lam/(1-(alpha/math.e)*(1.-lam))+(1.-p)*lam)

def sample_unif(b, n):
    lambdas = np.random.uniform(0.,b,n)
    return lambdas

def sample_logit_norm(n):
    lambdas = np.random.randn(n)
    lambdas = [math.exp(k)/(1+math.exp(k)) for k in lambdas]
    return lambdas

def sample_normal(n):
    mu, sigma = 0.5, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, n)
    '''
    abs(mu - np.mean(s)) < 0.01
    abs(sigma - np.std(s, ddof=1)) < 0.01
    count, bins, ignored = plt.hist(s, 30, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
            np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')
    plt.show()
    '''
    return s

def get_plot_unif(beta = 0.9, p = 0.9, n = 10000):
    alphas = np.linspace(0,math.e, 250)
    t = len(alphas)
    rev_avgs = np.zeros((t,1))
    br = [0.5]
    vr = [0.1, 0.2, 0.3]
    legend_list = []
    for b in br:
        for v in vr:
            for j in range(t):
                lambdas = sample_unif(b, n)
                alpha = alphas[j]
                k = math.e/(alpha*(1.-beta))
                rev_avgs[j] = sum([comp_rev(k, alpha, v, lam, p, beta) for lam in lambdas])/n
            plt.plot(alphas, rev_avgs)
            legend_list.append("b={0}, v={1}".format(b, v))
    plt.xlabel("Proportionality Constant Value")
    plt.ylabel("Rate of Revenue of A")
    plt.legend(legend_list, loc='upper left')
    plt.show()

def get_alpha_dist(b, p, v, beta, n, dist):
    alphas = np.linspace(0,math.e, 250)
    t = len(alphas)
    rev_avgs = np.zeros((t,1))
    for j in range(t):
        if dist == "unif":
            lambdas = sample_unif(b, n)
        elif dist == "logit":
            lambdas = sample_logit_norm(n)
        elif dist == "normal":
            lambdas = sample_normal(n)
        alpha = alphas[j]
        k = math.e/(alpha*(1.-beta))
        rev_avgs[j] = sum([comp_rev(k, alpha, v, lam, p, beta) for lam in lambdas])/n
    return(alphas, rev_avgs)

def run_main(b = 0.5, p = 0.9, v = 0.1, beta = 0.9, n = 10000, dist="unif"):
    (ualphas, urev_avgs) = get_alpha_dist(b, p, v, beta, n, dist)
    plt.plot(ualphas, urev_avgs)
    (lalphas, lrev_avgs) = get_alpha_dist(b, p, v, beta, n, dist="logit")
    plt.plot(lalphas, lrev_avgs)
    (nalphas, nrev_avgs) = get_alpha_dist(b, p, v, beta, n, dist="normal")
    plt.plot(nalphas, nrev_avgs)
    plt.xlabel("Proportionality Constant Value")
    plt.ylabel("Rate of Revenue of A")
    plt.legend(["Uniform Distribution","Logit Normal Distribution", 
        "Normal Distribution"], loc='upper left')
    plt.show()

'''
b = 0.5
p = 0.9
v = 0.1
beta = 0.9
n = 10000

plot_alpha_dist(p, v, beta, n, dist="logit")
plot_alpha_dist(p, v, beta, n, dist="unif")
'''
