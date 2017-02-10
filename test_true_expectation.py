import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def rev_rate_thresh(k, lam, beta, R, delta, t0, p):
	# compute revenue rate with a threshold function
	# with prob p, look-ahead = t1 >= delta
	# with prob 1-p, look-ahead = t0 < delta
	# reward R after k purchaces
	# discouting factor beta and exogenous prob lam

	j1 = range(int(delta), int(100*delta))
	f1 = [nCr(j-1,delta-1)*(lam**delta)*((1-lam)**(j-delta))*(k-R)/(j+k-delta) for j in j1]
	i0 = float(k-t0)
	j2 = range(int(i0), int(100*i0))
	f2 = [nCr(j-1,i0-1)*(lam**i0)*((1-lam)**(j-i0))*(k-R)/(j+k-i0) for j in j2]

	return p*sum(f1)+(1-p)*sum(f2)

def compute_delta(v, R, beta):
	# compute delta (related to phase transition) for 
	# duopoly where reward is R, discouting is beta and 
	# price difference is v

	return np.floor(np.log(v/(R*beta*(1.-beta)))/np.log(beta))


v = 0.1
lam = 0.05
beta = 0.9
R = 20.*v

delta = compute_delta(v, R, beta)
print delta

#p = np.random.rand()
p = 0.9
t0 = 0

k = range(int(delta), int(20*delta))

rev_rate = [rev_rate_thresh(j, lam, beta, R, delta, t0, p) for j in k]

f1 = plt.figure()
plt.plot(k, rev_rate)
plt.xlabel('k')
plt.ylabel('revenue rate')
plt.show()
# print solve_thresh(lam, R, delta, t0, p)

# r = np.linspace((1-lam)*t0,(1-lam)*float(delta))
# root = [max(solve_thresh(lam, j, delta, t0, p)) for j in r]

# f2 = plt.figure()
# plt.plot(r, root)
# plt.xlabel('R')
# plt.ylabel('max critical point')

# plt.show()