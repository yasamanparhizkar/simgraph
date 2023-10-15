%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **Frank-Wolfe iterations when Node i's color is blue
%
% author: Yasaman Parhizkar
% email me any questions: ypar@yorku.ca
% date: April 21st, 2023
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [M,...
    delta,...
    scaled_M,...
    scaled_factors,...
    M_current_eigenvector,...
    min_objective,...
    bins,...
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
    remaining_idx,...
    lu_bound_idx,...
    options,...
    exitflag] = LP_iterations_blue_yasaman(current_node,league_vec,league_vec_temp,flip_number,...
    scaled_factors_h,...
    Ms_off_diagonal,...
    feature_N,...
    G,...
    M,...
    delta,...
    V,...
    node_number,...
    remaining_idx,...
    M_current_eigenvector,...
    rho,...
    gamma,...
    beta,...
    C,...
    scaled_M,...
    scaled_factors,...
    bins,...
    objective_previous,...
    nv,...
    n_delta,...
    delta_idx,...
    data_feature,...
    num_list,...
    options,...
    LP_A_sparse_i,...
    LP_A_sparse_j,...
    LP_A_sparse_s,...
    LP_b,...
    LP_lb,...
    LP_ub,...
    dia_idx,...
    zz,...
    max_iter,...
    FW_dia_offdia_tol)

tol_offdia=Inf;

M_temp_best=M;

objective_previous_temp=objective_previous;

ddd = (0 - rho - (sum(abs(Ms_off_diagonal),2)-diag(abs(Ms_off_diagonal))));
sign_vecdd = flip_number'*current_node*-1;

% color = [current_node; flip_number];
% sign_vec = -1*color*color.';

LP_A_sparse_s(1:feature_N-1)=sign_vecdd.*abs(scaled_factors_h);

LP_A_sparse_j(feature_N)=feature_N-1+node_number;

scaler_v = abs(scaled_factors(remaining_idx,node_number));

for LP_A_i=1:feature_N-1
    temp_index=feature_N+(LP_A_i-1)*2+1;
    temp_index1=feature_N+(LP_A_i-1)*2+2;
    LP_A_sparse_s(temp_index)=sign_vecdd(1,LP_A_i)*scaler_v(LP_A_i);
    LP_A_sparse_j(temp_index1)=feature_N-1+remaining_idx(LP_A_i);
    LP_b(LP_A_i+1)=ddd(LP_A_i);
end

temp_index=feature_N-1+1+2*(feature_N-1);

for LP_A_i=1:n_delta
    temp_index1=temp_index+(LP_A_i-1)*(2*feature_N);

    % more efficient implementation is possible for the following block
    f_i=data_feature(delta_idx(LP_A_i,1),:);
    f_j=data_feature(delta_idx(LP_A_i,2),:);
    f_l=data_feature(delta_idx(LP_A_i,3),:);
    F_ijl=((f_i-f_j).')*(f_i-f_j) - ((f_i-f_l).')*(f_i-f_l);
    LP_A_sparse_s(temp_index1+1:temp_index1+feature_N-1)=2*F_ijl(node_number,remaining_idx);
    FM = M(remaining_idx, remaining_idx) .* F_ijl(remaining_idx, remaining_idx);
    dddd = sum(FM,"all")-sum(diag(FM),"all");
    LP_b(feature_N+LP_A_i)=-gamma-dddd(1,1);

end

LP_A = sparse(LP_A_sparse_i,LP_A_sparse_j,LP_A_sparse_s,feature_N+n_delta,feature_N-1+feature_N+n_delta);

LP_lb(sign_vecdd==-1)=-Inf;
LP_ub(sign_vecdd==-1)=0;
LP_lb(sign_vecdd==1)=0;
LP_ub(sign_vecdd==1)=Inf;

zero_mask=ones(2*feature_N-1,1);
lu_bound_idx = 1; % just a dummy - Yasaman
% lu_bound_idx=scaler_v==0;
% LP_lb(lu_bound_idx)=0;
% LP_ub(lu_bound_idx)=0;
% zero_mask(lu_bound_idx)=0;

LP_Aeq = [];
LP_beq = [];

% remove trace constraint by Yasaman
% LP_b(feature_N+1)=[];
% LP_A(6,:) = [];

%% LP settings end
[s_k,fval,exitflag,output,lambda] = linprog(G,...
    LP_A,LP_b,...
    LP_Aeq,LP_beq,...
    LP_lb,LP_ub,options);

%% debug by Yasaman
%inequalities = (LP_A * s_k);
%constraints = inequalities <= LP_b;
% disp(['Constraints are satisfied: ' num2str(all(constraints))])
% M_yas = full(scaled_M);
% M_yas(1:n_feature+1:end)=0;
% cond_yas = diag(full(scaled_M)) - sum(abs(M_yas),2);
% disp(' ');
% disp(['node_number = ' num2str(node_number)]);
% disp(['min objective value = ' num2str(obj_f)]);
% disp(['minimal eigenvalue of M = ' num2str(min(eig(M)))]);
% disp(['PD conditions satisfied: ' num2str(all(cond_yas >= rho-eps))]);
% % disp(['PD conditions (must be ge 0): ' num2str(cond_yas.' - rho)]);
% disp(['PD conditions unsatisfied: ' num2str(cond_yas(cond_yas - rho < 0).' - rho)]);
% % disp(['first eigenvector = ' num2str(fv1.') ]);
% disp(['first eigenvector has 0 entries: ' num2str(any((fv1 > -eps) .* (fv1 < eps)))]);
% disp(['exitflag = ' num2str(exitflag)]);

while isempty(s_k) == 1
    disp('===trying with larger OptimalityTolerance===');
    options.OptimalityTolerance = options.OptimalityTolerance*10;
    options.ConstraintTolerance = options.ConstraintTolerance*10;
    fprintf('===new OptimalityTolerance: %d===\n',options.OptimalityTolerance);
    [s_k,fval,exitflag,output,lambda] = linprog(G,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
    %inequalities = (LP_A * s_k);
    %constraints = inequalities <= LP_b;
end
%% set a step size
delta_updated = s_k(2*feature_N:end);

if isequal(league_vec,league_vec_temp)==1
    t_M21 = s_k(1:2*feature_N-1).*zero_mask;
    
    M_updated=M_temp_best;
    M_updated(node_number,remaining_idx)=t_M21(1:feature_N-1);
    M_updated(remaining_idx,node_number)=M_updated(node_number,remaining_idx);
    M_updated(dia_idx)=t_M21(feature_N-1+1:end);
else
    M21_updated = s_k(1:2*feature_N-1).*zero_mask;
    
    M_updated = M_temp_best;
    M_updated(remaining_idx,node_number)=M21_updated(1:feature_N-1);
    M_updated(node_number,remaining_idx)=M_updated(remaining_idx,node_number);
    M_updated(dia_idx)=M21_updated(feature_N-1+1:end);
    
    %=replace the following block if you run SGML on a different
    %objective function from GLR=======================================
    %[ L_c ] = graph_Laplacian( partial_sample, c, M_updated );% replace this if you need to run SGML on a different objective function
    %min_objective = x' * L_c * x;% replace this if you need to run SGML on a different objective function
    min_objective = sum(M_updated .* (V.'), 'all')+beta*sum(delta_updated,"all");
    %==================================================================
    
    %% reject the result (reject the color change) if it is larger than previous
    if min_objective>=objective_previous_temp
        min_objective=objective_previous_temp;
        %disp('color update return');
        return
        %% there is no need to iterate, since the node color is changed
    else
        M_temp_best = M_updated;
        delta = delta_updated;
        %disp('color update break');
        % break % no need to iterate, not even once, otherwise it is wrong.
    end
end

%% evaluate the objective value
%=replace the following block if you run SGML on a different
%objective function from GLR=======================================
%[ L_c ] = graph_Laplacian( partial_sample, c, M_updated );% replace this if you need to run SGML on a different objective function
%min_objective = x' * L_c * x;% replace this if you need to run SGML on a different objective function
min_objective = sum(M_updated .* (V.'), 'all')+beta*sum(delta_updated,"all");
%======================================================================

if min_objective>=objective_previous_temp
    min_objective=objective_previous_temp;
    %disp('early stop');
    return
end

M_temp_best = M_updated;
delta = delta_updated;

%% choose the M_temp_best that has not been thresholded to compute the gradient

%=replace the following gradient function if you need to run SGML 
%on a different objective function=====================================
% Yasaman: Extra step; we already had the correct G as input
[ G ] = compute_coefficient( feature_N, V, nv, n_delta, node_number, remaining_idx, zz, beta );
%======================================================================

tol_offdia=norm(min_objective-objective_previous_temp);

objective_previous_temp=min_objective;

%M_temp_best(abs(M_temp_best)<1e-5)=0;

%% detect subgraphs
bins_temp=bins;
M_current_eigenvector0=M_current_eigenvector;
num_list0=num_list;
% if sum(abs(M_temp_best(node_number,remaining_idx)))==0 % disconnected
%     if feature_N==max(bins_temp) % already disconnected
%     else
%         bins_temp(node_number)=max(bins_temp)+1; % assign a subgraph number
%         M_current_eigenvector0(num_list0==node_number)=[]; % heuristicaly remove the 1st entry of M_current_eigenvector as the lobpcg warm start
%         num_list0(num_list0==node_number)=[];
%         M_current_eigenvector0=M_current_eigenvector0/sqrt(sum(M_current_eigenvector0.^2));
%     end
% end
%% evaluate the temporarily accepted result with temporary scaled_M and scaled_factors

[M_current_eigenvector0,scaled_M_,scaled_factors_] = scalars(M_temp_best,feature_N,1,M_current_eigenvector0,bins_temp);

% lower_bounds = sum(abs(scaled_M_),2)-abs(scaled_M_(dia_idx))+rho;

%% reject the result if the lower_bounds are larger than C
% % Yasaman: Delete this if you delete the trace constraint
% if sum(lower_bounds) > C
%     min_objective=objective_previous;
%     %disp(['lower bounds sum:' num2str(sum(lower_bounds))]);
%     %disp('========lower bounds sum larger than C!!!========');
%     return
% end

%% M_temp_best passes all tests, now update the results
bins=bins_temp;
M=M_temp_best;
scaled_M=scaled_M_;
scaled_factors=scaled_factors_;
M_current_eigenvector=M_current_eigenvector0;
num_list=num_list0;
end

