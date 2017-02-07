v = 0.1;
p = 0.9;
beta = 0.9;

comp_delta = @(k, alpha) max(0,-log(alpha*k*(1-beta))/log(beta));

comp_rev = @(delta, l, k, alpha) (1-alpha*v)*(p*k*l/(k-(1-l)*delta)+(1-p)*l);

alpha_range = 0:0.01:exp(1);

trials = length(alpha_range);

rev_avgs = zeros(trials,1);

n = 10000;

for kk = 1:trials,
    alpha = alpha_range(kk);
    
    x = randn(n,1);
    lam = exp(x)./(1+exp(x));

%     b = 0.5;
%     lam = b*rand(n,1);

    k = exp(1)/(alpha*(1-beta));
    
    delta = comp_delta(k, alpha);

    rev_tot = 0;
    for j = 1:n,
        rev_tot = rev_tot+comp_rev(delta, lam(j), k, alpha);
    end

    rev_avgs(kk) = rev_tot/n;
end

% p = 0.9;
% comp_delta = @(k, alpha) max(0,-log(alpha*k*(1-beta))/log(beta));
% comp_rev = @(delta, l, k, alpha) (1-alpha*v)*(p*k*l/(k-(1-l)*delta)+(1-p)*l);

figure()
plot(alpha_range, rev_avgs)
hold on

for kk = 1:trials,
    alpha = alpha_range(kk);
    
%     x = randn(n,1);
%     lam = exp(x)./(1+exp(x));
 
    b = 1;
    lam = b*rand(n,1);

    k = exp(1)/(alpha*(1-beta));
    
    delta = comp_delta(k, alpha);

    rev_tot = 0;
    for j = 1:n,
        rev_tot = rev_tot+comp_rev(delta, lam(j), k, alpha);
    end

    rev_avgs(kk) = rev_tot/n;
end

%plot(alpha_range, rev_avgs, 'r')
ylabel('RoR_A')
xlabel('alpha')
%legend('b = 0.5', 'b = 1')
title('v = 0.1, p = 0.5')