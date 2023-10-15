function [C_minus] = sdcut_c_minus(u, A, B)
%EQ10_CALC_C_MINUS Summary of this function goes here
%   Detailed explanation goes here
m = length(B);
C = -1 * A;
for i = 1 : m
    C = C + u(i) * B{i};
end
[v, d] = eig(C);
idx_minus=find(diag(d)<0);
C_minus=v(:,idx_minus) * d(idx_minus,idx_minus) * v(:,idx_minus)';
C_minus=(C_minus+C_minus')/2;
end

