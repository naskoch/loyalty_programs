import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def rev_rate_thresh(k, lam, v, beta, R, delta, t0, p):
	# compute revenue rate with a threshold function
	# with prob p, look-ahead = t1 >= delta
	# with prob 1-p, look-ahead = t0 < delta
	# reward R after k purchaces
	# discouting factor beta and exogenous prob lam

	return (lam*p*(k-R))/(k-(1.-lam)*delta)+(lam*(1.-p)*(k-R))/(k-(1.-lam)*t0)

def compute_delta(v, R, beta):
	# compute delta (related to phase transition) for 
	# duopoly where reward is R, discouting is beta and 
	# price difference is v

	return max(0.,np.floor(np.log(v/(R*(1.-beta)))/np.log(beta)))

v = 0.1
beta = 0.9
p = 0.5

lam = 0.9
alpha = np.linspace(0.5,3.)
#k = range(1,100)
k = np.array([math.e/(alpha[j]*(1.-beta)) for j in range(len(alpha))])
R = [alpha[j]*v*k[j] for j in range(len(alpha))]

delta = [compute_delta(v, j, beta) for j in R]

revs = [rev_rate_thresh(k[j], lam, v, beta, R[j], delta[j], 0, p) for j in range(len(k))]

f1 = plt.figure()
plt.plot(alpha, revs)
plt.xlabel('alpha')
plt.ylabel('rev rate')

plt.show()