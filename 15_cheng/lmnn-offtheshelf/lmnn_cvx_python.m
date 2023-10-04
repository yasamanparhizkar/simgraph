function [M, delta, cvx_status] = lmnn_cvx_python(data_path, idstr,gamma, beta)
% set up for the script
close all; clc;
if exist('data_feature', 'var') == 1
	clear data_feature;
end
if exist('data_label', 'var') == 1
	clear data_label;
end
if exist('data_edge', 'var') == 1
	clear data_edge;
end

tic
%% Load the dataset from disc
load([data_path,'data_',idstr,'.mat'],'data_feature','data_label','data_edge');
data_feature = cast(data_feature, 'double');
data_label = data_label.';
[n_sample,n_feature] = size(data_feature);
data_edge = data_edge + 1;
toc

%% Train the model
%=Set parameters===========================================================
gamma = cast(gamma, 'double');
beta = cast(beta, 'double');
tic
[V] = get_objective_variables_ready(data_feature, data_label, data_edge, n_sample, n_feature);
toc
%==========================================================================
% find number of delta optimization variables
tic
[n_delta,delta_idx] = get_delta_variables(data_label, n_sample, data_edge);
disp(['n_delta = ', num2str(n_delta)]);
toc

tic
cvx_begin sdp
cvx_solver SeDuMi
variable M(n_feature,n_feature) symmetric semidefinite
variable delta(n_delta,1) nonnegative
minimize(sum(sum(M.*V)) + beta*sum(delta))
subject to
for r=1:n_delta
    f_i = data_feature(delta_idx(r,1),:);
    f_j = data_feature(delta_idx(r,2),:);
    f_l = data_feature(delta_idx(r,3),:);
    f_ij = f_i - f_j;
    f_il = f_i - f_l;
    F_ijl = (f_ij.')*f_ij - (f_il.')*f_il;
    delta(r) - sum(sum(M.*F_ijl)) >= gamma;
    %delta(r) >= 0;
end
%delta >= sum(sum(M.*F_ijl)) + gamma;
%delta >= zeros(n_delta, 1);
%M.' == M;
%M>=0;
cvx_end
M = full(M);
toc
end

