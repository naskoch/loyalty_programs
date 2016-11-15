import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rev_rate_thresh(k, lam, beta, R, delta, t0, p):
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

	return np.floor(np.log(v/(R*beta*(1.-beta)))/np.log(beta))

def solve_thresh(lam, R, delta, t0, p):
	# solve for critical points of rev_rate_thresh function

	val1 = R-(1.-lam)*float(delta)
	val2 = R-(1.-lam)*t0

	a = p*val1+(1.-p)*val2
	b = -2*(1.-lam)*(p*val1*t0+(1.-p)*val2*float(delta))
	c = ((1.-lam)**2)*(p*val1*(t0**2)+(1.-p)*val2*(float(delta)**2))

	return np.roots([a, b, c])

v = 0.1
beta = 0.8
R = 30.*v

delta = compute_delta(v, R, beta)
print delta

#p = np.random.rand()
p = 0.1
t0 = 0

lam = p*(delta-R)/(delta-(delta-R)*(1.-p))*2.5
print lam
print rev_rate_thresh(delta, lam, beta, R, delta, t0, p)

k = np.linspace(delta,1.5*delta)
rev_rate = [rev_rate_thresh(j, lam, beta, R, delta, t0, p) for j in k]

f1 = plt.figure()
plt.plot(k, rev_rate)
plt.xlabel('k')
plt.ylabel('revenue rate')

print solve_thresh(lam, R, delta, t0, p)

print (lam*p*(R-(1.-lam)*delta))/((lam*delta)**2)+(lam*(1.-p)*R)/(delta**2)

print (1.-lam)*delta
print ((1.-lam)*(1.-p)*R*delta)/(p*(R-(1.-lam)*delta)+(1.-p)*R)
print delta

r = np.linspace((1-lam)*t0,(1-lam)*float(delta))
root = [max(solve_thresh(lam, j, delta, t0, p)) for j in r]

f2 = plt.figure()
plt.plot(r, root)
plt.xlabel('R')
plt.ylabel('max critical point')

plt.show()