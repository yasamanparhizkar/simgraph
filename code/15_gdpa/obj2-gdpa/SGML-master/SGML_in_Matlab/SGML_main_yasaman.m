%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **SGML main function
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 16th, 2020
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [M] = SGML_main_yasaman(M,...
    data_feature,...
    data_label,...
    n_sample,...
    n_feature,...
    tol_NR,...
    tol_GD,...
    rho,...
    max_iter,...
    GS_or_NR,...
    tol_golden_search,...
    options,...
    FW_dia_offdia_tol,...
    FW_full_tol,...
    C,...
    gamma,...
    nv_od,...
    nv_full,...
    zz,...
    dia_idx,...
    num_list,...
    league_vec,...
    bins,...
    fv1,...
    scaled_M,...
    scaled_factors,...
    LP_A_sparse_i,...
    LP_A_sparse_j,...
    LP_A_sparse_s,...
    LP_b,...
    LP_lb,...
    LP_ub)

%%=====objective function used here: graph Laplacian regularizer(GLR)======
% [c,y] = get_graph_Laplacian_variables_ready(data_feature,data_label,n_sample,n_feature); % replace this if you need to run SGML on a different objective function
% [ L ] = graph_Laplacian( n_sample, c, M ); % replace this if you need to run SGML on a different objective function
% initial_objective = data_label' * L * data_label; % replace this if you need to run SGML on a different objective function
[V] = get_objective_variables_ready(data_feature,data_label,n_sample,n_feature,gamma);
initial_objective = sum(M .* (V.'), 'all');
%%=========================================================================

disp(['initial objective value = ' num2str(initial_objective)]);

obj_f=initial_objective;

for node_number = 1:n_feature % optimzing the diagonals and one row/column of off-diagonals

    remaining_idx=1:n_feature;
    remaining_idx(node_number)=[];
    
    % compute gradient 
    % replace the following gradient function for GLR if you are using a different
    % objective function.
    [ G ] = compute_coefficient( n_feature, V, nv_od, node_number, remaining_idx );

    %% BLUE or RED starts
    Ms_off_diagonal = scaled_M(remaining_idx,remaining_idx); % the submatrix M22 after removing the node_number-th row/column of M.
    scaled_factors_h = scaled_factors(node_number,remaining_idx); % the scaled factors of M21
    
    %% try BLUE league on NODE node_number
    league_vec_temp = league_vec;
    league_vec_temp(node_number) = 1; % BLUE is 1
    league_vec_remaining = league_vec_temp;
    league_vec_remaining(node_number) = [];
    
    %% LP iterations BLUE
    [M_blue,...
        scaled_M_blue,...
        scaled_factors_blue,...
        fv1_blue,...
        min_obj_blue,...
        bins_blue,...
        num_list_blue,...
        LP_A_sparse_i,...
        LP_A_sparse_j,...
        LP_A_sparse_s,...
        LP_b,...
        LP_lb,...
        LP_ub,...
        LP_Aeq,...
        LP_beq,...
        zero_mask,...
        scaler_v,...
        remaining_idx,...
        lu_bound_idx,...
        options_blue,...
        exitflag_blue] = LP_iterations_blue_yasaman(1,...
        league_vec,...
        league_vec_temp,...
        league_vec_remaining,...
        scaled_factors_h,...
        Ms_off_diagonal,...
        n_feature,...
        G,...
        M,...
        V,...
        node_number,...
        remaining_idx,...
        fv1,...
        rho,...
        C,...
        scaled_M,...
        scaled_factors,...
        bins,...
        obj_f,...
        nv_od,...
        num_list,...
        options,...
        LP_A_sparse_i,...
        LP_A_sparse_j,...
        LP_A_sparse_s,...
        LP_b,...
        LP_lb,...
        LP_ub,...
        dia_idx,...
        max_iter,...
        FW_dia_offdia_tol);
    
    %% try RED league on NODE node_number
    league_vec_temp(node_number) = -1; % RED is -1
    
    %% LP iterations RED
    [M_red,...
        scaled_M_red,...
        scaled_factors_red,...
        fv1_red,...
        min_obj_red,...
        bins_red,...
        num_list_red,...
        exitflag_red] = LP_iterations_red_yasaman(-1,...
        league_vec,...
        league_vec_temp,...
        league_vec_remaining,...
        scaled_factors_h,...
        n_feature,...
        G,...
        M,...
        V,...
        node_number,...
        remaining_idx,...
        fv1,...
        rho,...
        C,...
        scaled_M,...
        scaled_factors,...
        bins,...
        obj_f,...
        nv_od,...
        num_list,...
        LP_A_sparse_i,...
        LP_A_sparse_j,...
        LP_A_sparse_s,...
        LP_b,...
        LP_lb,...
        LP_ub,...
        LP_Aeq,...
        LP_beq,...
        zero_mask,...
        scaler_v,...
        lu_bound_idx,...
        options,...
        dia_idx,...
        max_iter,...
        FW_dia_offdia_tol);
    
    if min_obj_red <= min_obj_blue
        M = M_red;
        scaled_M = scaled_M_red;
        scaled_factors = scaled_factors_red;
        fv1 = fv1_red;
        league_vec(node_number) = -1;
        bins = bins_red;
        num_list=num_list_red;
        obj_f=min_obj_red;
        exitflag = exitflag_red;
    end
    if min_obj_red > min_obj_blue
        M = M_blue;
        scaled_M = scaled_M_blue;
        scaled_factors = scaled_factors_blue;
        fv1 = fv1_blue;
        league_vec(node_number) = 1;
        bins = bins_blue;
        num_list=num_list_blue;
        obj_f=min_obj_blue;
        exitflag = exitflag_blue;
    end
    %% BLUE or RED ends

    %% debug by Yasaman
%     M_yas = full(scaled_M);
%     M_yas(1:n_feature+1:end)=0;
%     cond_yas = diag(full(scaled_M)) - sum(abs(M_yas),2);
%     disp(' ');
%     disp(['node_number = ' num2str(node_number)]);
%     disp(['min objective value = ' num2str(obj_f)]);
%     disp(['minimal eigenvalue of M = ' num2str(min(eig(M)))]);
%     disp(['PD conditions satisfied: ' num2str(all(cond_yas >= rho-eps))]);
%     % disp(['PD conditions (must be ge 0): ' num2str(cond_yas.' - rho)]);
%     disp(['PD conditions unsatisfied: ' num2str(cond_yas(cond_yas - rho < 0).' - rho)]);
%     % disp(['first eigenvector = ' num2str(fv1.') ]);
%     disp(['first eigenvector has 0 entries: ' num2str(any((fv1 > -eps) .* (fv1 < eps)))]);
%     disp(['exitflag = ' num2str(exitflag)]);
    
end

% compute gradient
% replace the following gradient function for GLR if you are using a different
% objective function.
[ G ] = compute_coefficient( n_feature, V, nv_full, 0, 0 );

[M] = LP_iterations_full_M_yasaman(league_vec,...
    scaled_factors,...
    n_feature,...
    G,...
    M,...
    V,...
    rho,...
    obj_f,...
    C,...
    nv_full,...
    zz,...
    options,...
    dia_idx,...
    max_iter,...
    FW_full_tol);

current_objective = sum(M .* (V.'), 'all');
disp(' ');
disp(['converged objective value = ' num2str(current_objective)]);
disp(['minimal eigenvalue of M = ' num2str(min(eig(M)))]);
end