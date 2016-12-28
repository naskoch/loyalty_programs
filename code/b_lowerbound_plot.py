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

n = 200

p = np.linspace(0,1,n)
v = np.linspace(1./math.e,0,n)

B = np.zeros((n,n))

for i in range(n):
	for j in range(n):
		B[i,j] = max(0., 2.*((1.-v[i])-p[j]/(1.-p[j])*(1.-math.e*v[i]))/((1.-v[i])+(1.-math.e*v[i])))
		B[i,j] = min(1., B[i,j])

fig, axes = plt.subplots(2,1)
heatmap = axes[0].imshow(B, cmap = 'viridis', extent=[0,1,0,1./math.e])


B = np.zeros((n,n))

for i in range(n):
	for j in range(n):
		B[i,j] = min(1., 2*p[j]/(p[j]+math.e*v[i]/(1.-math.e*v[i])))

heatmap = axes[1].imshow(B, cmap = 'viridis', extent=[0,1,0,1./math.e])

fig.colorbar(heatmap, ax=axes.ravel().tolist())

plt.xlabel(r'$\boldsymbol{p}$')
#plt.ylabel(r'$\boldsymbol{v}$')
fig.text(0.04, 0.5, r'$\boldsymbol{v}$', va='center', rotation='vertical')

plt.savefig('../report/figures/b_bounds.png')