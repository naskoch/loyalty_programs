import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# this program makes a plot for that
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

n = 200

p_range = np.linspace(0,1,n)
v_range = np.linspace(1./math.e,0,n)

x_range = []
y_range = []

for p in p_range:
    for v in v_range:
        alpha = math.e-math.sqrt((math.e*(1.-math.e*v*p))/(v*(1.-p)))
        if alpha < 1. and alpha > 0.:
            x_range.append(p)
            y_range.append(v)

plt.plot(x_range, y_range)

plt.xlabel(r'$\boldsymbol{p}$')
plt.ylabel(r'$\boldsymbol{v}$')

plt.savefig('alpha_approx.png')