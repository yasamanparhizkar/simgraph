%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **FW iterations on FULL M when Node i's colors are fixed
%
% author: Yasaman Parhizkar
% email me any questions: ypar@yorku.ca
% date: April 21st, 2023
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [M] = LP_iterations_full_M(league_vec,...
    scaled_factors,...
    feature_N,...
    G,...
    M,...
    V,...
    rho,...
    objective_previous,...
    S_upper,...
    nv,...
    zz,...
    options,...
    dia_idx,...
    max_iter,...
    FW_full_tol)

tol_full=Inf;
M_best_temp=M;
objective_previous_temp=objective_previous;

remaining_idx=0;
BCD=0;

%% LP settings start
scaled_factors__ = scaled_factors';
scaled_factors__(dia_idx)=[];
scaled_factors__ = reshape(scaled_factors__,feature_N-1,feature_N)';

total_offdia = sum(1:feature_N-1);

LP_A_sparse_i=zeros(1,feature_N^2+feature_N);
LP_A_sparse_j=LP_A_sparse_i;
LP_A_sparse_s=LP_A_sparse_i;

LP_A_sparse_i(1:feature_N)=1;
LP_A_sparse_j(1:feature_N)=total_offdia+1:total_offdia+feature_N;
LP_A_sparse_s(1:feature_N)=1;

LP_b = [S_upper;zeros(feature_N,1)-rho];

LP_lb=zeros(total_offdia+feature_N,1);
LP_ub=zeros(total_offdia+feature_N,1)+Inf;
LP_lb(total_offdia+1:end)=rho;

t_counter=0;
t_counter_c=0;
sign_vec=zeros(1,total_offdia);
sign_idx=zeros(feature_N);
scaled_factors_zero_idx = zeros(1,total_offdia);
for vec_i=1:feature_N
    for vec_j=1:feature_N
        if vec_j>vec_i
            t_counter=t_counter+1;
            if league_vec(vec_i)==league_vec(vec_j) % positive edge, negative m entry <0
                sign_vec(t_counter)=-1;
                LP_lb(t_counter)=-Inf;
                LP_ub(t_counter)=0;
            else
                sign_vec(t_counter)=1;
                LP_lb(t_counter)=0;
                LP_ub(t_counter)=Inf;
            end
            if scaled_factors(vec_i,vec_j)==0
                scaled_factors_zero_idx(t_counter)=1;
            end
        end
        t_counter_c=t_counter_c+1;
        sign_idx(vec_i,t_counter_c)=t_counter;
    end
    t_counter_c=0;
end

scaled_factors_zero_idx=logical(scaled_factors_zero_idx);
LP_lb(scaled_factors_zero_idx)=0;
LP_ub(scaled_factors_zero_idx)=0;

zero_mask=ones(total_offdia+feature_N,1);
zero_mask(scaled_factors_zero_idx)=0;

sign_idx=triu(sign_idx,1);
sign_idx=sign_idx+sign_idx';
sign_idx(dia_idx)=[];
sign_idx=reshape(sign_idx,feature_N-1,feature_N)';

for LP_i=1:feature_N
    temp_index=feature_N+(LP_i-1)*feature_N+1:feature_N+LP_i*feature_N-1;
    temp_index1=feature_N+LP_i*feature_N;
    LP_A_sparse_i(temp_index)=1+LP_i;
    LP_A_sparse_i(temp_index1)=1+LP_i;
    LP_A_sparse_j(temp_index)=sign_idx(LP_i,:);
    LP_A_sparse_j(temp_index1)=total_offdia+LP_i;
    LP_A_sparse_s(temp_index)=abs(scaled_factors__(LP_i,:)).*sign_vec(sign_idx(LP_i,:));
    LP_A_sparse_s(temp_index1)=-1;
end

LP_A=sparse(LP_A_sparse_i,LP_A_sparse_j,LP_A_sparse_s,1+feature_N,total_offdia+feature_N);

LP_Aeq = [];
LP_beq = [];

% remove trace constraint by Yasaman
% LP_b(1)=[];
% LP_A(1,:) = [];

%% LP settings end
net_gc=[2*G(zz);diag(G)];

[s_k,fval,exitflag,output,lambda] = linprog(net_gc,...
    LP_A,LP_b,...
    LP_Aeq,LP_beq,...
    LP_lb,LP_ub,options);

%% debug by Yasaman
% inequalities = (LP_A * s_k);
% constraints = inequalities <= LP_b;

%% ===Gurobi Matlab interface might be faster than Matlab linprog======
% you need to apply an Academic License (free) in order to use Gurobi Matlab
% interface: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
% once you have an Academic License and have Gurobi Ooptimizer
% installed, you should be able to run the following code by
% uncommenting them.
%     s_k = gurobi_matlab_interface(net_gc,...
%        LP_A,LP_b,...
%        LP_Aeq,LP_beq,...
%        LP_lb,LP_ub,options);
%======================================================================

while isempty(s_k) == 1
    disp('===trying with larger OptimalityTolerance===');
    options.OptimalityTolerance = options.OptimalityTolerance*10;
    options.ConstraintTolerance = options.ConstraintTolerance*10;
    [s_k,fval,exitflag,output,lambda] = linprog(net_gc,...
        LP_A,LP_b,...
        LP_Aeq,LP_beq,...
        LP_lb,LP_ub,options);
    
    %% ===Gurobi Matlab interface might be faster than Matlab linprog==
    % you need to apply an Academic License (free) in order to use Gurobi Matlab
    % interface: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
    % once you have an Academic License and have Gurobi Ooptimizer
    % installed, you should be able to run the following code by
    % uncommenting them.
%         s_k = gurobi_matlab_interface(net_gc,...
%            LP_A,LP_b,...
%            LP_Aeq,LP_beq,...
%            LP_lb,LP_ub,options);
    %==================================================================
    
end

%% proximal gradient to determine the Frank-Wolfe step size starts
t_M21 = s_k.*zero_mask;
M_updated=M_best_temp;
M_updated(zz)=t_M21(1:total_offdia);
M_updated_t=M_updated';
M_updated(zz')=M_updated_t(zz');

M_updated(dia_idx)=t_M21(total_offdia+1:end);

%% evaluate the objective value

%=replace the following block if you run SGML on a different
%objective function from GLR=======================================
%[ L_c ] = graph_Laplacian( partial_sample, c, M_updated );
min_objective = sum(M_updated .* (V.'), 'all');
%======================================================================

if min_objective>=objective_previous_temp
    M=M_best_temp;
    return
end

M_best_temp = M_updated;

%% choose the M_best_temp that has not been thresholded to compute the gradient

%=replace the following block if you run SGML on a different
%objective function from GLR=======================================
[ G ] = compute_coefficient( feature_N, V, nv, BCD, remaining_idx );
%======================================================================

tol_full=norm(min_objective-objective_previous_temp);

objective_previous_temp=min_objective;

M=M_best_temp;
end

