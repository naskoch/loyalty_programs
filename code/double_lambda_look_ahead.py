import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rev_rate_thresh(k, lam_a, lam_b, beta, R, delta, t0, p):
	# compute revenue rate with a threshold function
	# with prob p, look-ahead = t1 >= delta
	# with prob 1-p, look-ahead = t0 < delta
	# reward R after k purchaces
	# discouting factor beta and exogenous prob lam

	return (lam_b*p*(k-R))/(k-(1.-lam_b*(1+lam_a))*delta)+(lam_b*(1.-p)*(k-R))/(k-(1.-lam_b*(1+lam_a))*t0)

def compute_delta(v, R, beta, lam_a, lam_b):
	# compute delta (related to phase transition) for 
	# duopoly where reward is R, discouting is beta and 
	# price difference is v

	return max(0.,np.floor(np.log((v*(1.-lam_a*beta))/(R*beta*(1.-beta)))/np.log((1.-lam_a)*beta/(1.-lam_a*beta))))

v = 0.5
beta = 0.8
R = 1.
lam_a = 0.05
lam_b = 0.05

delta = compute_delta(v, R, beta, lam_a, lam_b)
print delta

#p = np.random.rand()
p = 0.5
t0 = 0

k = np.linspace(delta,10.*delta+10)
rev_rate = [rev_rate_thresh(j, lam_a, lam_b, beta, R, delta, t0, p) for j in k]

f1 = plt.figure()
plt.plot(k, rev_rate)
plt.xlabel('k')
plt.ylabel('revenue rate')

plt.show()