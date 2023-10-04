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
function [n_delta,delta_idx] = get_delta_variables(x,N,edges)
delta_idx = [];
for edge_i=1:size(edges,1)
    n_i = edges(edge_i,1);
    n_j = edges(edge_i,2);
    if x(n_i) == x(n_j)
        n_ls = [edges(edges(:,2)==n_i,1);edges(edges(:,1)==n_i,2)];
        for li = 1:size(n_ls,1)
            if x(n_ls(li)) == -x(n_i)
                delta_idx = [delta_idx;n_i,n_j,n_ls(li)];
            end
        end
    end
end
n_delta=size(delta_idx,1);
end

