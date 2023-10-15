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
function [ G ] = compute_coefficient( n, V, k, BCD, remaining_idx )
if k==n+(n*(n-1)/2) && BCD==0% full
    G=V.';
elseif k==2*n-1 % one row/column of off-diagonals and diagonals
    G=[V(BCD, remaining_idx).'; diag(V)];
elseif k==n % diagonals
    G=diag(V);
else % one row/column of off-diagonals
    G=V(BCD, remaining_idx).';
end
end



