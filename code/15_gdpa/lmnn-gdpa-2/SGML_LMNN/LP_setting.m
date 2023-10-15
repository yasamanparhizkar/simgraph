%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **get parts of linear constraints ready for Matlab linprog
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 16th, 2020
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [LP_A_sparse_i,...
    LP_A_sparse_j,...
    LP_A_sparse_s,...
    LP_b,...
    LP_lb,...
    LP_ub] = LP_setting(n_feature,n_delta,rho,data_feature,delta_idx)

% get the parts of linear constraints ready for Matlab linprog
% the following parts of linear constraints do not have to be updated in
% each iteration of optimization.

% \sum_{j|j\neq i}|s_i^t M_{i,j}/s_j^t|+rho \leq M_{i,i}^t
% the above line means: the sum of the absolute values of the scaled
% off-diagonals should be smaller than the diagonal.

% M_{i,1} \leq 0, if i \in N_b
% the above line means: positive edge weight (negative M_{i,1}) if Node i is blue.

% M_{i,1} \geq 0, if i \in N_r
% the above line means: negative edge weight (positive M_{i,1}) if Node i is red.

% \sum_i M_{i,i} \leq C
% the above line means: trace of M is not larger than C

LP_A_sparse_i=zeros(1,n_feature-1+1+2*(n_feature-1)+n_delta*2*n_feature); % total number of entries in LP_A that needs to be defined
LP_A_sparse_j=LP_A_sparse_i;
LP_A_sparse_s=LP_A_sparse_i;

LP_A_sparse_i(1:n_feature-1)=1;
LP_A_sparse_j(1:n_feature-1)=1:n_feature-1;
LP_A_sparse_i(n_feature)=1;
LP_A_sparse_j(n_feature)=n_feature;
LP_A_sparse_s(n_feature)=-1;

LP_b = zeros(n_feature+n_delta,1);
LP_b(1)=-rho; % \sum_{j|j\neq i}|s_i^t M_{i,j}/s_j^t|+rho \leq M_{i,i}^t
% therefore, \sum_{j|j\neq i}|s_i^t M_{i,j}/s_j^t| - M_{i,i}^t \leq -rho


for LP_A_i=1:n_feature-1
    
    temp_index=n_feature+(LP_A_i-1)*2+1;
    temp_index1=n_feature+(LP_A_i-1)*2+2;
    LP_A_sparse_i(temp_index)=LP_A_i+1;
    LP_A_sparse_j(temp_index)=LP_A_i;
    LP_A_sparse_i(temp_index1)=LP_A_i+1;
    %LP_A_sparse_j(temp_index1)=n_feature+LP_A_i;
    LP_A_sparse_s(temp_index1)=-1;
    
end

temp_index=n_feature-1+1+2*(n_feature-1);

for LP_A_i=1:n_delta
    
    temp_index1=temp_index+(LP_A_i-1)*(2*n_feature);
    LP_A_sparse_i(temp_index1+1:temp_index1+2*n_feature-1)=n_feature+LP_A_i;
    LP_A_sparse_i(temp_index1+2*n_feature)=n_feature+LP_A_i;
    LP_A_sparse_j(temp_index1+1:temp_index1+2*n_feature-1)=1:(2*n_feature-1);
    LP_A_sparse_j(temp_index1+2*n_feature)=2*n_feature-1+LP_A_i;
    LP_A_sparse_s(temp_index1+2*n_feature)=-1;
    
    % more efficient implementation is possible for the following block
    f_i=data_feature(delta_idx(LP_A_i,1),:);
    f_j=data_feature(delta_idx(LP_A_i,2),:);
    f_l=data_feature(delta_idx(LP_A_i,3),:);
    F_ijl=((f_i-f_j).')*(f_i-f_j) - ((f_i-f_l).')*(f_i-f_l);
    LP_A_sparse_s(temp_index1+n_feature:temp_index1+2*n_feature-1)=diag(F_ijl).';

end

LP_lb=zeros(n_feature-1+n_feature+n_delta,1); % lower bounds of the variables.
% because we are optimizing the diagonals and one row/column of off-diagonals,
% the number of variables is [n_feature-1+n_feature].
LP_ub=zeros(n_feature-1+n_feature+n_delta,1); % upper bounds of the variables.
% because we are optimizing the diagonals and one row/column of off-diagonals,
% the number of variables is [n_feature-1+n_feature].
LP_lb(n_feature-1+1:n_feature-1+n_feature)=rho; % M_{i,i} \geq 0 + rho
LP_ub(n_feature-1+1:n_feature-1+n_feature+n_delta)=Inf; % M_{i,i} \leq Inf

end

