%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **compute objective funtion's gradient
%
% author:Yasaman Parhizkar
% email me any questions: ypar@yorku.ca
% date: April 21st, 2023
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [ G ] = compute_coefficient( n, V, k, n_delta, BCD, remaining_idx, zz, beta )
if k==n+(n*(n-1)/2) && BCD==0% full
    temp=V.';
    G=[2*temp(zz);diag(temp);beta*ones([n_delta 1])];
elseif k==2*n-1 % one row/column of off-diagonals and diagonals
    G=[2*V(BCD, remaining_idx).'; diag(V);beta*ones([n_delta 1])];
elseif k==n % diagonals
    G=[diag(V); beta*ones([n_delta 1])];
else % one row/column of off-diagonals
    G=[2*V(BCD, remaining_idx).';beta*ones([n_delta 1])];
end
end



