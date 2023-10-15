%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **get some GLR objective function's variable ready
%
% author: Yasaman Parhizkar
% email me any questions: ypar@yorku.ca
% date: April 21th, 2023
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [V] = get_objective_variables_ready(feature,x,N,n)
y=(x-x.').^2;
y=reshape(y,[N^2 1]);
a=reshape(feature,[N 1 n]);
c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
V = c.' * ((1-y/4) .* c);
end

