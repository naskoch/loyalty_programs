import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import simulation_lambda_from_dist as sim

# FIGURES FOR EQUAL BUDGET

# plot percentage of extraneous trips needed as a 
# function of beta

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

f1 = plt.figure()
p = 0.5
b = 0.4
beta = 0.9
alpha = np.linspace(0.5,2.5)
v = 0.01
k = np.array([math.e/(alpha[j]*(1.-beta)) for j in range(len(alpha))])
delta = np.array([-1.*math.log(alpha[j]*k[j]*(1.-beta))/math.log(beta) for j in range(len(k))])
rev_rate = np.array([p*k[j]*(1.-alpha[j]*v)*(1./((delta[j]**2)*b)*(delta[j]*b-(k[j]-delta[j])*math.log((k[j]-delta[j]*(1.-b))/(k[j]-delta[j]))))+(1.-p)*(1.-alpha[j]*v)*b/2 for j in range(len(k))])

plt.plot(alpha, (k-delta)/k)
plt.plot(alpha, rev_rate)
#plt.plot(alpha, delta, label=r'$\Delta$')
plt.xlabel(r'$\alpha$')
plt.legend()

plt.show()

f2 = plt.figure()
b = np.linspace(0,1)
lhs = np.array([1./b[j]*(1.-(math.e-1.)/b[j]*math.log(1.+b[j]/(math.e-1.))) for j in range(len(b))])

v = 0.1
rhs = np.array([(1.-(1.-b[j])*(1.-v))/(2*b[j]*math.e*(1.-v)) for j in range(len(b))])

plt.plot(b, lhs)
plt.plot(b, rhs)
#plt.xlabel()
#plt.ylabel()

plt.savefig('b_plot.pdf')