v = 0.1;

lam = 0.05;
beta = 0.9;

R = v*20;

del = floor(log(v/(R*beta*(1-beta)))/log(beta));

%t0 = 0;
%t1 = del+1;

% threshold

p = rand();
% revenue rate for threshold function (as a function of k and t0) given all
% other values
f_thresh = @(k, t0) (lam*p*(k-R))./(k-(1-lam)*del)+(lam*(1-p)*(k-R))./(k-(1-lam)*t0);

% interval of k's to look at
k = 1:2*del;

% (discrete) uniform random look ahead on interval [0, T] - not necessary
% for first problem

T = 2*del;
obj_vec = @(k,t) (lam*(k-R))./((k-(1-lam)*t)*T);
f_uni = @(k) (lam*(k-R)*(T-del))./((k-(1-lam)*del)*T)+sum(obj_vec(k,0:1:(del-1)));

uni = zeros(length(k),1);
for i = 1:length(k),
    uni(i) = f_uni(k(i));
end

figure()
plot(k,f_thresh(k, 0))
hold on
plot(k,f_thresh(k, del-1),'r')
plot(k, uni, 'g')
legend('t0 = 0', 't0 = del-1', 'uniform')
xlabel('k')
ylabel('Revenue rate')

t0 = 0;
disp(solve_thresh(p,R,lam,del,t0))
% c1 = R-(1-lam)*del;
% c2 = R-(1-lam)*t0;
% disp(c1)
% disp(c2)