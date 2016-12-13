import numpy as np
import math
import matplotlib.pyplot as plt

def ror_A(k, R, v, b, p, beta, delta):
    #delta = compute_delta(v, R, beta)
    lval = p*(k-R) / (b*delta**2) * (b*delta - (k-delta)*np.log(1+b*delta/(k-delta)))
    rval = (1-p)*b*(k-R)/(2*k)
    return lval + rval

def ror_B(k, R, v, b, p, beta, delta):
    #delta = compute_delta(v, R, beta)
    lval = p*(k-delta)*(1-v) / (b*delta**2) * (k*np.log(1 + b*delta/(k-delta)) - b*delta)
    rval = (1-p)*(1-b/2)*(1-v)
    return lval + rval

def compute_delta(v, R, beta):
	return max(0.,np.floor(np.log(v/(R*(1.-beta)))/np.log(beta)))

def pbpairplot(alpha=1):
    v = 0.05 
    beta = 0.95
    k = math.e/(alpha * (1-beta)) 
    R = alpha * k * v
    delta = max(0., np.floor(-1.0/np.log(beta)))
    p_range = [0.0001*j for j in range(1,10000)]

    b_range = [0.0001*j for j in range(1,10000)] 

    x_range = []
    y_range = []

    for b in b_range:
        for p in p_range:
            rA = ror_A(k, R, v, b, p, beta, delta)
            rB = ror_B(k, R, v, b, p, beta, delta)
            if rA > rB:
                x_range.append(b)
                y_range.append(p)

    plt.plot(x_range, y_range)
    plt.ylabel("p value")
    plt.xlabel("b value")
    plt.show()
