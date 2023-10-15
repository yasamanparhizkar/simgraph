function [U] = ...
    Cholesky_M( feature_train_test, ...
    initial_label_index, ...
    class_train_test)

tol_main=1e-5;

partial_feature = feature_train_test(initial_label_index,:);
partial_observation = class_train_test(initial_label_index);
[partial_sample,n_feature]=size(partial_feature);
S_upper=n_feature;

run_t=1;
time_vec=zeros(run_t,1);
obj_vec=zeros(run_t,1);
time_eig=zeros(run_t,1);
max_iter=1e3;
for time_i=1:run_t
    
    tStart=tic;
    %     M=initial_M(n_feature,1);
    U=initial_U(n_feature,1);
    [c,~] = get_graph_Laplacian_variables_ready(partial_feature,partial_observation,partial_sample,n_feature);
    lr=.1/partial_sample;%finding a good choice of a step size is out of the scope of the paper
    
    counter_diag_nondiag = 0;
    
    tol_diag_nondiag = Inf;
    
    while tol_diag_nondiag > tol_main
        
        if counter_diag_nondiag == 0
            
            %[L] = graph_Laplacian( partial_sample, c, M );
            [ L ] = Cholesky_Decomposition_graph_Laplacian( partial_sample, c, U);
            initial_objective = partial_observation' * L * partial_observation;
            disp(['initial = ' num2str(initial_objective)]);
            
        end
        
        [ G ] = Matrix_U_compute_gradient_autograd(U,partial_observation,partial_sample,c);
        U_updated = U - lr * extractdata(G);
        U_updated = tril(U_updated);
%         U_updated(:,6:end) = 0;
        
        if sum(diag(U_updated*U_updated'))>S_upper
            factor_for_diag = sum(diag(U_updated*U_updated'))/S_upper;
            U_updated = U_updated/sqrt(factor_for_diag);
        end
        
        %         [ M_updated,...
        %             time_eig] = PDcone_projection(...
        %             partial_sample,...
        %             n_feature,...
        %             c,...
        %             y,...
        %             M,...
        %             S_upper,...
        %             time_eig,...
        %             time_i,...
        %             lr);
        
        %         [ L ] = graph_Laplacian( partial_sample, c, M_updated );
        [ L ] = Cholesky_Decomposition_graph_Laplacian( partial_sample, c, U_updated);
        
        current_objective = partial_observation' * L * partial_observation;
        
        if current_objective>initial_objective
            lr=lr/2;
        else
            lr=lr*(1+1e-2);
        end
        
        %         M = M_updated;
        U = U_updated;
        
        tol_diag_nondiag = norm(current_objective - initial_objective);
        
        initial_objective = current_objective;
        
        counter_diag_nondiag = counter_diag_nondiag + 1;
        if counter_diag_nondiag==max_iter
            break
        end
        
    end
    disp(['converged = ' num2str(current_objective)]);
    
    time_vec(time_i)=toc(tStart);
    obj_vec(time_i)=current_objective;
end

% disp(['time_vec mean: ' num2str(mean(time_vec)) ' std:' num2str(std(time_vec))]);
% disp(['obj_vec mean: ' num2str(mean(obj_vec)) ' std:' num2str(std(obj_vec))]);
end