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

def ror_A_opt_k(p, alpha, b, v):
    forward_ror = (1-alpha*v) * (p * math.e/alpha * (1 - (math.e-alpha)/(b*alpha) * math.log(1 + b*alpha/(math.e-alpha))))
    backward_ror = (1-alpha*v) * (1-p)*b/2
    return forward_ror + backward_ror

def ror_B_opt_k(p, alpha, b, v):
    forward_ror = (1-v) * (p * math.e/alpha * ((math.e-alpha)/(b*alpha) * math.log(1 + b*alpha/(math.e-alpha)) - (math.e-alpha)/math.e))
    backward_ror = (1-v) * (1-p) * (1-b/2)
    return forward_ror + backward_ror

def pbpairplot_optk(alpha=1):
    v = 0.05 
    beta = 0.95
    p_range = [0.001*j for j in range(1,1000)]

    b_range = [0.001*j for j in range(1,1000)] 

    x_range = []
    y_range = []

    for b in b_range:
        for p in p_range:
            rA = ror_A_opt_k(p, alpha, b, v)
            rB = ror_B_opt_k(p, alpha, b, v)
            if rA > rB:
                x_range.append(b)
                y_range.append(p)

    plt.plot(x_range, y_range)
    plt.ylabel("p value")
    plt.xlabel("b value")
    plt.show()

def pbpairplot(alpha=1):
    v = 0.05 
    beta = 0.95
    k = math.e/(alpha * (1-beta)) 
    R = alpha * k * v
    #delta = max(0., np.floor(-1.0/np.log(beta)))
    delta = 1./(1-beta)
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

def pbpairplot_rorA(file, alpha=1):
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
            if rA > b/2.:
                x_range.append(b)
                y_range.append(p)

    plt.plot(x_range, y_range)
    plt.ylabel("p value")
    plt.xlabel("b value")
    plt.savefig(file)

def pbpairplot_both(file, alpha=1):
    v = 0.05
    beta = 0.95
    k = math.e/(alpha * (1-beta)) 
    R = alpha * k * v
    delta = max(0., np.floor(-1.0/np.log(beta)))
    p_range = [0.0001*j for j in range(1,10000)]

    b_range = [0.0001*j for j in range(1,10000)] 

    x_range_1 = []
    y_range_1 = []

    x_range_2 = []
    y_range_2 = []

    x_range_3 = []
    y_range_3 = []

    for b in b_range:
        for p in p_range:
            rA = ror_A(k, R, v, b, p, beta, delta)
            rB = ror_B(k, R, v, b, p, beta, delta)
            if rA > rB and rA > b/2.:
                x_range_3.append(b)
                y_range_3.append(p)
            elif rA > rB:
                x_range_1.append(b)
                y_range_1.append(p)
            elif rA > b/2.:
                x_range_2.append(b)
                y_range_2.append(p)

    plt.plot(x_range_1, y_range_1, 'b')
    plt.plot(x_range_2, y_range_2, 'y')
    plt.plot(x_range_3, y_range_3, 'g')
    plt.ylabel("p value")
    plt.xlabel("b value")
    plt.savefig(file)