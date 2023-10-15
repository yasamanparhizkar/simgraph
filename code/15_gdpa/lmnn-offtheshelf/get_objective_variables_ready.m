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
function [V] = get_objective_variables_ready(feature,x,edges,N,n)
V = zeros(n,n);
for edge_i=1:size(edges,1)
    n_i = edges(edge_i, 1);
    n_j = edges(edge_i, 2);
    if x(n_i) == x(n_j)
        f_i = feature(n_i,:);
        f_j = feature(n_j,:);
        fij = f_i - f_j;
        V = V + (fij.') * fij;
    end
end
end

