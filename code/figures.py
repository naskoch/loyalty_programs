import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import simulation_lambda_from_dist as sim

# FIGURES FOR EQUAL BUDGET

# plot percentage of extraneous trips needed as a 
# function of beta

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# f = plt.figure()
# beta = np.linspace(0.01,0.99)
# k = math.e/(1.-beta)
# delta = [-1.*math.log(k[j]*(1.-beta[j]))/math.log(beta[j]) for j in range(len(k))]

# plt.plot(beta, (k-delta)/k)
# plt.xlabel(r'$\beta$')
# plt.ylabel(r'$(k-\Delta)/k$')

# plt.savefig('phase_trans.pdf')

# Plot from simulations

T = 1000 # time period of interest
n = 1000 # number of customers

# define threshold look-ahead distribution
#p = 0.5
t0 = 0
t1 = T

# monopoly effects of each form (can also model
# preferences)
# each person has a random lambda from in (0, b)
bs = np.linspace(0,1,20)

# other parameters
beta = 0.9

f, ax = plt.subplots(2,3, sharex='col', sharey='row')

trials = 10

# fixed prices of firms
#vs = [[0.01, 0.05, 0.1], [0.2, 0.3, 0.4]]

# look ahead distribution
ps = [[0.25, 0.5, 0.75], [0.8, 0.9, 1.]]

print ax
v = 0.2
# this affects proportion of budgets
alpha = 2.5
for i in range(2):
	for j in range(3):
		p = ps[i][j]
		print p
		avg_rev_a = np.zeros((len(bs),1))
		avg_rev_b = np.zeros((len(bs),1))
		avg_tot_R = np.zeros((len(bs),1))
		for l in range(len(bs)):
			b = bs[l]
			s_a = 0.
			s_b = 0.
			s_R = 0.
			k = np.floor(math.e/(alpha*(1.-beta)))
			#k = np.floor(math.e**(t1*(1.-beta))/(1.-beta))
			R = alpha*k*v
			delta = sim.compute_delta(v, R, beta)
			for jj in range(trials):
				rev_a, rev_b, tot_a, tot_b, tot_R = sim.sim_period(T, n, p, t0, t1, b, v, R, delta, k)
				s_a += rev_a
				s_b += rev_b
				s_R += tot_R
			avg_rev_a[l] = s_a/float(trials)
			avg_rev_b[l] = s_b/float(trials)
			avg_tot_R[l] = s_R/float(trials)

		(ax[i,j]).plot(bs, avg_rev_a/(T*n), label = 'B')
		(ax[i,j]).plot(bs, avg_rev_b/(T*n), 'r', label = 'A')
		#ax[i][j].xlabel('b')
		#ax[i][j].ylabel('rev')
		(ax[i,j]).legend()
		(ax[i,j]).set_title('p = %.2f' %p)

f.text(0.5, 0.04, 'b', ha='center')
f.text(0.04, 0.5, 'rev', va='center', rotation='vertical')

plt.savefig('test.pdf')