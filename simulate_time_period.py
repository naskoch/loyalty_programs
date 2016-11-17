import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Firm A has traditional pricing
# Firm B has frequency reward program

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

def compute_delta(v, R, beta, lam_a, lam_b):
	# compute delta (related to phase transition) for 
	# duopoly where reward is R, discouting is beta and 
	# price difference is v

	return np.floor(np.log((v*(1.-lam_a*beta))/(R*beta*(1.-beta)))/np.log((1.-lam_a)*beta/(1.-lam_a*beta)))

def phase_vec(t, delta, k):
	# compute i0 for each customer
	i0 = np.zeros((n,1))
	for j in range(len(t)):
		if t[j] < delta:
			i0[j] = k-t[j]
		else:
			i0[j] = k-delta
	return i0

def sim_one_purchase(V, i0, lam_a, lam_b, k):
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

	P = np.zeros((n,1))
	for j in range(len(V)):
		q = np.random.rand()
		if q <= lam_a:
			# wp lambda_a, must purchase from a
			P[j] = 0
			num_a += 1
		elif q <= lam_a+lam_b:
			# wp lambda_b, must purchase from a
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

def sim_period(T, n, p, t0, t1, lam_a, lam_b, v, R, delta, k):
	# simulate entire period of programs, length T

	# sample customers
	t = sample_look_ahead(t0, t1, p, n)

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
	for k in range(T):
		V, P, num_a, num_b, num_R = sim_one_purchase(V, i0, lam_a, lam_b, k)
		# update revenues
		rev_a += (1.-v)*float(num_a)
		rev_b += float(num_b)-(R*float(num_R))
		tot_a += num_a
		tot_b += num_b
		tot_R += num_R

	return rev_a, rev_b, tot_a, tot_b, tot_R

T = 100 # time period of interest
n = 1000 # number of customers

# define threshold look-ahead distribution
p = 0.5
t0 = 0
t1 = 100

# monopoly effects of each form (can also model
# preferences)
lam_a = 0.05
lam_b = 0.05

# other parameters
beta = 0.8
v = 0.25

# setup reward program
R = 1.
delta = compute_delta(v, R, beta, lam_a, lam_b)
k = delta

rev_a, rev_b, tot_a, tot_b, tot_R = sim_period(T, n, p, t0, t1, lam_a, lam_b, v, R, delta, k)

print delta
print k-R

print rev_a
print rev_b

print tot_a
print tot_b
print tot_R