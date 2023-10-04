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
function [V] = get_objective_variables_ready_python(swap_path, idstr)
load([swap_path,'data_',idstr,'.mat'], 'data_feature','data_label','data_edge');
data_feature = cast(data_feature, 'double');
data_label = data_label.';
data_edge = data_edge + 1;
[n_sample,n_feature] = size(data_feature);

V = zeros(n_feature,n_feature);
for edge_i=1:size(data_edge,1)
    n_i = data_edge(edge_i, 1);
    n_j = data_edge(edge_i, 2);
    if data_label(n_i) == data_label(n_j)
        f_i = data_feature(n_i,:);
        f_j = data_feature(n_j,:);
        fij = f_i - f_j;
        V = V + (fij.') * fij;
    end
end
end

