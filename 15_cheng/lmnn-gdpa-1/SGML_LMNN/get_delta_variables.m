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
function [n_delta,delta_idx] = get_delta_variables(x,N)
y=(x-x.').^2;
z=(1-(y/4 ))-eye(N);
w=y/4;
z=reshape(z, [N N 1]);
w=reshape(w, [N 1 N]);
zw=z .* w;
n_delta=sum(zw,'all');

%find indices of delta variables
idx=reshape(1:N^3, [N N N]);
idx=idx(logical(zw));
idx_i=mod(idx-1, N)+1;
idx_j=mod(floor((idx-1)/N),N)+1;
idx_l=floor((idx-1)/(N^2))+1;
delta_idx=[idx_i,idx_j,idx_l];
end

