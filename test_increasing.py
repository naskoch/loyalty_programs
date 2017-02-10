import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

v = 0.3
beta = 0.9
R = 1.
lam_a = 0.
lam_b = 0.95

p = 0.5
t0 = 0

k = 10

V = np.zeros((k,1))
V[k-1] = R
for i in range(k-2,-1,-1):
	a = ((1.-lam_a-lam_b)*v+lam_b*beta*V[i+1])/(1.-(1.-lam_b)*beta)
	print a
	b = ((1.-lam_a)*beta*V[i+1])/(1.-lam_a*beta)
	print b
	V[i] = max(a,b)

print V

f1 = plt.figure()
plt.plot(range(k), V)
plt.show()