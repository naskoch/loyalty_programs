function x = solve_thresh(p, R, lam, del, t0)

val1 = R-(1-lam)*del;
val2 = R-(1-lam)*t0;

a = p*val1+(1-p)*val2;
b = -2*(1-lam)*(p*val1*t0+(1-p)*val2*del);
c = (1-lam)^2*(p*val1*t0^2+(1-p)*val2*del^2);

x = roots([a;b;c]);