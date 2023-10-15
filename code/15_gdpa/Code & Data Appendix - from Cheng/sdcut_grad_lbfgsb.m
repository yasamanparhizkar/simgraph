function [obj,grad] = sdcut_grad_lbfgsb(u,A,B,b,sigma)
%EQ10_CALC_DUAL_OBJ_GRAD_LBFGSB Summary of this function goes here
%   Detailed explanation goes here

%% compute C_minus
C_minus=sdcut_c_minus(u,A,B);

%% objective
obj = (-0.25 / sigma) * sum(C_minus(:) .^ 2)  - u' * b;
obj = -1 * obj;

%% gradient
m = length(u);
grad = zeros(m, 1);
for ii = 1 : m
    grad(ii) = C_minus(:)' * B{ii}(:);
end
grad = (0.5 / sigma) * grad + b;
end

