import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Assume alpha -> e, then we have a simple condition
# on b to have RoR_A > RoR_B

# this program makes a plot for that
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

p = np.linspace(0,1)
e = math.e
v = 0.05

UB = [min(1.,2*j/(j+e*v/(1.-e*v))) for j in p]
LB = [min(1.,max(0.,2*((1.-v)-j/(1.-j)*(1.-e*v))/((1.-v)+(1.-e*v)))) for j in p]

plt.plot(UB, p, label='upper bound', linewidth=2.0)
plt.plot(LB, p, label='lower bound', linewidth=2.0)
#plt.xlim([0,1])
#plt.ylim([0,2])
plt.legend()

plt.xlabel(r'$\boldsymbol{b}$')
plt.ylabel(r'$\boldsymbol{p}$')

#plt.show()
plt.savefig('../report/figures/b_region_v3.png')