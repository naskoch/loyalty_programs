import numpy as np
import math

def ror_A(k, R, v, b, p, beta):
    delta = compute_delta(v, R, beta)
    lval = p*(k-R) / (b*delta**2) * (b*delta - (k-delta)*np.log(1+b*delta/(k-delta)))
    rval = (1-p)*b*(k-R)/(2*k)
    return lval + rval

def ror_B(k, R, v, b, p, beta):
    delta = compute_delta(v, R, beta)
    lval = p*(k-delta)*(1-v) / (b*delta**2) * (k*np.log(1 + b*delta/(k-delta)) - b*delta)
    rval = (1-p)*(1-b/2)*(1-v)
    return lval + rval

def compute_delta(v, R, beta):
	return max(0.,np.floor(np.log(v/(R*(1.-beta)))/np.log(beta)))

v = 0.2 
beta = 0.9
alpha = 1 
k = math.e/(alpha * (1-beta)) 
R = alpha * k * v

p_range = [0.1*j for j in range(1,10)]

b_range = [0.1*j for j in range(1,10)] 

ror_gap = [[ror_A(k, R, v, b, p, beta) - ror_B(k, R, v, b, p, beta) for b in b_range] for p in p_range]
