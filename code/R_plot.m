v = 0.1;
%R = v*20;

lam = 0.05;
beta = 0.9;

del = @(R) floor(log(v./(R*beta*(1-beta)))./log(beta));

R = 0:0.01:50;

figure()
plot(R,(1-lam)*del(R),'LineWidth',2)
hold on
plot(R,R,'r','LineWidth',2)
xlabel('R')
legend('(1-lambda)*delta','R')
% t0 = 2;
% plot(R,(1-lam)*t0*ones(length(R),1),'g','LineWidth',2)

% checking which values of v allow the quadratic (solution to threshold) to
% be solved for at least some R
v = 0:0.01:1;
int_check = zeros(length(v),1);
for k = 1:length(v),
    del = @(R) floor(log(v(k)./(R*beta*(1-beta)))./log(beta));
    R1 = 0:0.01:50;
    R2 = (1-lam)*del(R1);
    if min(R1-R2) < 0,
        % two things intersect
        int_check(k) = 1;
    end
end

figure()
plot(v,int_check)