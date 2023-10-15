clear all;
clc;
close all;
rng('default');

% cvx_precision low: [ϵ3/8,ϵ1/4,ϵ1/4]
% cvx_precision medium: [ϵ1/2,ϵ3/8,ϵ1/4]
% cvx_precision default: [ϵ1/2,ϵ1/2,ϵ1/4]
% cvx_precision high: [ϵ3/4,ϵ3/4,ϵ3/8]
% cvx_precision best: [0,ϵ1/2,ϵ1/4]

addpath(genpath('cdcs_implementation\'));
addpath('L-BFGS-B_implementation\');

cvx_begin quiet
cvx_precision low
cvx_precision
K_=[1e5 5e4 1e4 5e3 1e3 5e2 1e2 5e1 2e1];
results=zeros(9,10);
[dataset_str,read_data] = get_data_large(18);
for ki=1:9
    %%=======change the following things for experiments======
    
    rho=0; % PSD parameter for gdpa---align Gershgorin discs' left ends to 0's
    experiment_save_str=['results\results_' dataset_str '_min_max_scaling_large_aaai23.mat'];
    %%========================================================
    
    label=read_data(:,end);
    K=K_(ki);
    rng(0);
    indices = crossvalind('Kfold',label,K); % K-fold cross-validation
    for fold_i=1:1
        read_data_i=read_data(indices==fold_i,:);
        read_data_i=[read_data_i read_data_i(:,1)];
        read_data_i(:,1)=[];
        n_sample=size(read_data_i,1);
        for rsrng=1:1
            disp('==============================================================');
            disp(['======================= dataset ' num2str(18) ' fold ' num2str(fold_i) ' run number: ' num2str(rsrng) ' =======================']);
            disp('==============================================================');
            disp('==============================================================');
            b_ind = 1; % 1 training sample
            [cL,label] = dataLoader_normalized_minmax(read_data_i,n_sample,b_ind,rsrng);
            
            alpha=1e15;
            sw=0.5;
            u=1;
            
            if ki<=6
                %% SeDuMi
                disp('SeDuMi ====================================================================');
                clear dB dA;
                dL = [-cL zeros(n_sample,1); zeros(1,n_sample) 0];
                ei_s = eye(n_sample);
                for i=1:length(b_ind)
                    dB(:,:,i) = [zeros(n_sample,n_sample) ei_s(:,b_ind(i)); ei_s(b_ind(i),:) 0];
                end
                [x_pred_sedumi,error_count_sedumi,obj_sedumi,t_sedumi] = ...
                    sedumi_sdp_prime(dL,dB,n_sample,b_ind,label);
                disp(['SeDuMi error_count: ' num2str(error_count_sedumi)]);
                disp(['SeDuMi obj: ' num2str(obj_sedumi)]);
                
                %% MOSEK
                disp('MOSEK ====================================================================');
                [x_pred_mosek,error_count_mosek,obj_mosek,t_mosek] = ...
                    mosek_sdp_prime(dL,dB,n_sample,b_ind,label);
                disp(['MOSEK error_count: ' num2str(error_count_mosek)]);
                disp(['MOSEK obj: ' num2str(obj_mosek)]);
                
                %% CDCS 8 (sdp primal)
                disp('CDCS (8) ====================================================================');
                ei_l = eye(n_sample+1);   %% two identity matrices
                for i=1:n_sample+1
                    dA(:,:,i) = diag(ei_l(:,i)); % eq. (7), where dA is a stacked matrix where each of them has only 1 non-zero entry
                end
                [y_cdcs8,z_cdcs8,obj_cdcs8,db_cdcs8,x_pred_cdcs8,error_count_cdcs8,...
                    t_cdcs8,u_cdcs8,alpha_cdcs8] = cdcs8(label,b_ind,n_sample,dA,dB,dL);
                disp(['CDCS (8) error_count: ' num2str(error_count_cdcs8)]);
                disp(['CDCS (8) obj: ' num2str(obj_cdcs8)]);
                
                %% BCR
                disp('BCR ====================================================================');
                [error_count_bcr,t_bcr] = bcr(label,b_ind,n_sample,dA,dB,dL);
                disp(['BCR error_count: ' num2str(error_count_bcr)]);
                
                %% SDcut
                disp('SDcut ====================================================================');
                [y_sdcut,z_sdcut,obj_sdcut,db_sdcut,x_pred_sdcut,error_count_sdcut,...
                    t_sdcut,u_sdcut,alpha_sdcut] = sdcut(label,b_ind,n_sample,dA,dB,dL);
                disp(['SDcut error_count: ' num2str(error_count_sdcut)]);
                disp(['SDcut obj: ' num2str(obj_sdcut)]);
                
                %% CDCS (20)
                disp('CDCS (20)====================================================================');
                [obj_cdcs20,error_count_cdcs20,...
                    t_cdcs20] = cdcs20(label,b_ind,n_sample,dA,dB,dL,u,alpha,sw);
                disp(['CDCS (20) error_count: ' num2str(error_count_cdcs20)]);
                disp(['CDCS (20) obj: ' num2str(obj_cdcs20)]);
                
                %% GLR-box
                disp('GLR-box ====================================================================');
                [err_count_glrbox,obj_glrbox,t_glrbox] = ...
                    glr_box_constraints(cL,n_sample,b_ind,label);
                disp(['GLR-box error_count: ' num2str(err_count_glrbox)]);
                disp(['GLR-box obj: ' num2str(t_glrbox)]);                   
      
            end
            
            %% GDPA
            disp('GDPA ====================================================================');
            [obj_gdpa,x_pred_gdpa,err_count_gdpa,u00_gdpa,alpha00_gdpa,eigen_gap_gdpa,...
                t_gdpa] = ...
                gdpa_vec(label,b_ind,n_sample,cL,u,alpha,sw,...
                rho);
            disp(['GDPA error_count: ' num2str(err_count_gdpa)]);
            disp(['GDPA obj: ' num2str(obj_gdpa)]);

%             %% GLR
%             disp('GLR ====================================================================');
%             [err_count_glr,obj_glr,t_glr] = ...
%                 glr_closed_form(cL,n_sample,b_ind,label);
%             disp(['GLR error_count: ' num2str(err_count_glr)]);
%             disp(['GLR obj: ' num2str(t_glr)]);      

            %% SNS
            disp('SNS ====================================================================');
            [error_count_sns,obj_sns,t_sns] = sns(cL,n_sample,b_ind,label);
            disp(['SNS error_count: ' num2str(error_count_sns)]);
            disp(['SNS obj: ' num2str(t_sns)]); 

            if ki<=6
                results(ki,:)=[n_sample t_sedumi...
                    t_mosek...
                    t_cdcs8...
                    t_bcr...
                    t_sdcut...
                    t_cdcs20...
                    t_gdpa...
                    t_glrbox...
                    t_sns];
            else
                results(ki,:)=[n_sample 0 ...
                    0 ...
                    0 ...
                    0 ...
                    0 ...
                    0 ...
                    t_gdpa ...
                    0 ...
                    t_sns];
            end
            
        end
    end
    
    clearvars -except dataset_i experiment_save_str results K_ ki dataset_str read_data
    cvx_precision low;
end
results(:,2:10)=results(:,2:10)*1000;
save(experiment_save_str);