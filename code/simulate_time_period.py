import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Firm A has traditional pricing
# Firm B has frequency reward program

def rev_rate_thresh(k, lam_a, lam_b, beta, R, delta, t0, p):
	# compute revenue rate with a threshold function
	# with prob p, look-ahead = t1 >= delta
	# with prob 1-p, look-ahead = t0 < delta
	# reward R after k purchaces
	# discouting factor beta and exogenous prob lam

	return (lam_b*p*(k-R))/(k-(1.-lam_b*(1+lam_a))*delta)+(lam_b*(1.-p)*(k-R))/(k-(1.-lam_b*(1+lam_a))*t0)

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

	return max(0.,np.floor(np.log((v*(1.-lam_a*beta))/(R*beta*(1.-beta)))/np.log((1.-lam_a)*beta/(1.-lam_a*beta))))

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
v = 0.001

# setup reward program
R = 1.
delta = compute_delta(v, R, beta, lam_a, lam_b)
if rev_rate_thresh(delta, lam_a, lam_b, beta, R, delta, t0, p) > lam_a:
	k = delta
else:
	k = T

trials = 10
vs = np.linspace(0.25, 0.001)
avg_rev_a = np.zeros((len(vs),1))
avg_rev_b = np.zeros((len(vs),1))
for l in range(len(vs)):
	v = vs[l]
	s_a = 0.
	s_b = 0.
	for j in range(trials):
		rev_a, rev_b, tot_a, tot_b, tot_R = sim_period(T, n, p, t0, t1, lam_a, lam_b, v, R, delta, k)
		s_a += rev_a
		s_b += rev_b
	avg_rev_a[l] = s_a/float(trials)
	avg_rev_b[l] = s_b/float(trials)

f1 = plt.figure()
plt.plot(vs, avg_rev_a, label = 'A')
plt.plot(vs, avg_rev_b, 'r', label = 'B')
plt.xlabel('v')
plt.ylabel('avg rev')
plt.legend()

plt.show()