import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rev_rate_thresh(k, lam, v, beta, delta, p):
	# compute revenue rate with a threshold function
	# with prob p, look-ahead = t1 >= delta
	# with prob 1-p, look-ahead = t0 < delta
	# reward R after k purchaces
	# discouting factor beta and exogenous prob lam

	return (lam*p*(k-k*v))/(k-(1.-lam)*delta)+(lam*(1.-p)*(1.-v))

def compute_delta(v, R, beta):
	# compute delta (related to phase transition) for 
	# duopoly where reward is R, discouting is beta and 
	# price difference is v

	return max(0.,np.floor(np.log(v/(R*beta*(1.-beta)))/np.log(beta)))

def solve_thresh(lam, R, delta, t0, p):
	# solve for critical points of rev_rate_thresh function

	val1 = R-(1.-lam)*float(delta)
	val2 = R-(1.-lam)*t0

	a = p*val1+(1.-p)*val2
	b = -2*(1.-lam)*(p*val1*t0+(1.-p)*val2*float(delta))
	c = ((1.-lam)**2)*(p*val1*(t0**2)+(1.-p)*val2*(float(delta)**2))

	return np.roots([a, b, c])

# v = 0.35
# beta = 0.9
# p = 0.5

# lam = 0.4

# N = range(20,400)
# k = [v*j for j in N]

# delta = [compute_delta(v, j, beta) for j in N]

# revs = [rev_rate_thresh(k[j], lam, v, beta, delta[j], p) for j in range(len(N))]

# f1 = plt.figure()
# plt.plot(k, revs)
# plt.xlabel('k')
# plt.ylabel('rev rate')

# plt.show()

lam = 0.4
p = 0.4
R = 1.3
delta = 1.0

denom = delta-(delta-R)*(1.-p)
l = (p*(delta-R))/denom

print (l**2)*(1.-p)*R+p*l*delta-p*(delta-R)
print (-1.*p*(delta-R)*(1.-p)*(R**2))/(denom**2)