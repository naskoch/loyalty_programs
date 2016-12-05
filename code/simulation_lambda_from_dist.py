import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Firm A has traditional pricing
# Firm B has frequency reward program

def rev_rate_thresh(k, lam, beta, R, delta, t0, p):
	# compute revenue rate with a threshold function
	# with prob p, look-ahead = t1 >= delta
	# with prob 1-p, look-ahead = t0 < delta
	# reward R after k purchaces
	# discouting factor beta and exogenous prob lam

	return (lam*p*(k-R))/(k-(1.-lam)*delta)+(lam*(1.-p)*(k-R))/(k-(1.-lam)*t0)

def sample_look_ahead(t0, t1, p, n):
	# wp p, t = t1 and wp 1-p, t = t0
	t = np.zeros((n,1))
	q = np.random.rand(n,1)
	for k in range(n):
		if q[k] <= p:
			t[k] = t1
		else:
			t[k] = t0
	return t

def sample_lam(n, b):
	# sample a lambda for the n customers
	# uniform from (0,b)

	lam = [b*np.random.rand() for j in range(n)]
	return lam

def compute_delta(v, R, beta):
	# compute delta (related to phase transition) for 
	# duopoly where reward is R, discouting is beta and 
	# price difference is v

	return max(0.,np.floor(np.log(v/(R*(1.-beta)))/np.log(beta)))

def phase_vec(t, delta, k):
	# compute i0 for each customer
	i0 = np.zeros((len(t),1))
	for j in range(len(t)):
		if t[j] < delta:
			i0[j] = k-t[j]
		else:
			i0[j] = k-delta
	return i0

def sim_one_purchase(V, i0, lam, k):
	# simulate a single purchase for each customer
	# based on current state
	# return a vector of puchase decisions
	# if 0, customer purchases from A
	# if 1, customer purchases from B

	# also return number of purchase made to A, to B
	# and number of rewards given out - update state vector doing so

	# if customer collects a reward, we restart reward program

	num_a = 0
	num_b = 0
	num_R = 0

	P = np.zeros((len(V),1))
	for j in range(len(V)):
		q = np.random.rand()
		if q <= lam[j]:
			P[j] = 1
			if V[j] < k-1:
				V[j] += 1
				num_b += 1
			else:
				V[j] = 0
				num_R += 1
		else:
			# customer makes decision based on current state
			if V[j] < i0[j]:
				P[j] = 0
				num_a += 1
			else:
				P[j] = 1
				if V[j] < k-1:
					V[j] += 1
					num_b += 1
				else:
					V[j] = 0
					num_R += 1
	return V, P, num_a, num_b, num_R

def sim_period(T, n, p, t0, t1, b, v, R, delta, k):
	# simulate entire period of programs, length T

	# sample customers
	t = sample_look_ahead(t0, t1, p, n)
	lam = sample_lam(n, b)

	# create vector for states of each vector
	V = np.zeros((n,1))

	# compute phase transitions
	i0 = phase_vec(t, delta, k)

	rev_a = 0.
	rev_b = 0.

	# also keep track of total visits and total rewards
	tot_a = 0
	tot_b = 0
	tot_R = 0

	# simulate days and keep track of revenue
	for j in range(T):
		V, P, num_a, num_b, num_R = sim_one_purchase(V, i0, lam, k)
		# update revenues
		rev_a += (1.-v)*float(num_a)
		rev_b += float(num_b)-(R*float(num_R))
		tot_a += num_a
		tot_b += num_b
		tot_R += num_R

	return rev_a, rev_b, tot_a, tot_b, tot_R

if __name__ == "__main__":

	T = 1000 # time period of interest
	n = 1000 # number of customers

	# define threshold look-ahead distribution
	p = 0.5
	t0 = 0
	t1 = T

	# monopoly effects of each form (can also model
	# preferences)
	# each person has a random lambda from in (0, b)
	b = 0.9

	# other parameters
	beta = 0.9

	# fixed prices of firms
	v = 0.1

	# setup reward program, optimizing revenue rate of B
	# k = math.e/(1.-beta)
	# R = v*k
	# delta = compute_delta(v, R, beta)
	# if rev_rate_thresh(delta, lam, beta, R, delta, t0, p) > lam:
	# 	k = delta
	# else:
	# 	k = T

	trials = 20
	vs = np.linspace(0.25, 0.1, 5)
	avg_rev_a = np.zeros((len(vs),1))
	avg_rev_b = np.zeros((len(vs),1))
	avg_tot_R = np.zeros((len(vs),1))
	for l in range(len(vs)):
		v = vs[l]
		s_a = 0.
		s_b = 0.
		s_R = 0.
		k = np.floor(math.e/(1.-beta))
		R = k*v
		delta = compute_delta(v, R, beta)
		# if delta != 0. and rev_rate_thresh(delta, b/2, beta, R, delta, t0, p) > b/2:
		# 	# here we check based on average of lambda distribution
		# 	# need some theory to work this out, but testing for now
		# 	k = delta
		# else:
		# 	k = T
		for j in range(trials):
			rev_a, rev_b, tot_a, tot_b, tot_R = sim_period(T, n, p, t0, t1, b, v, R, delta, k)
			s_a += rev_a
			s_b += rev_b
			s_R += tot_R
		avg_rev_a[l] = s_a/float(trials)
		avg_rev_b[l] = s_b/float(trials)
		avg_tot_R[l] = s_R/float(trials)

	print avg_tot_R

	f1 = plt.figure()
	plt.plot(vs, avg_rev_a/(T*n), label = 'A')
	plt.plot(vs, avg_rev_b/(T*n), 'r', label = 'B')
	plt.xlabel('v')
	plt.ylabel('avg rev per person per day')
	plt.legend()

	plt.show()