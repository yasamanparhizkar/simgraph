function [obj_xlx,obj,err_count] = ...
    gdpa_tsp_main_uc_(label,b_ind,n_sample,L,dy,dz,epsilon)

%% LP settings
options=optimoptions('linprog','Algorithm','interior-point','display','none'); % linear program (LP) setting for Frank-Wolfe algorithm
% options=optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
options.OptimalityTolerance=1e-5; % LP optimality tolerance
options.ConstraintTolerance=1e-5; % LP interior-point constraint tolerance

% db = 2*label(b_ind); % training labels x 2
% dz_plus_idx = db < 0;
% dz_minus_idx = db > 0;
% dz_ind_plus = b_ind(dz_plus_idx);
% dz_ind_minus = b_ind(dz_minus_idx);

% db = 2*label(b_ind); % training labels x 2
db = zeros(length(label(b_ind)),1);
% dz_plus_idx = db < 0;
% dz_minus_idx = db > 0;
% dz_ind_plus = b_ind(dz_plus_idx);
% dz_ind_minus = b_ind(dz_minus_idx);
dz_ind_plus = 1;
dz_ind_minus = 1;
% epsilon=0;
%% initialize a psd matrix matrix H bar

y00=zeros(n_sample,1)+1e1;
% y00(b_ind)=1e0;
z_init=-(db/2)*0;
ynplus1=sum(y00)*1e-2;
y_init=[y00;ynplus1];
% epsilon=(sum(ynplus1)+sum(z_init));
% epsilon=sum(abs(z_init))*1e0;
y_init=sum(abs(L),2);
y_init=[y_init(2:end); y_init(1)];
% z_init=dz;
% epsilon=0;

% [initial_H] = construct_H_tsp(n_sample,...
%     -L(2:end,2:end),...
%     epsilon,...
%     y_init,...
%     z_init,...
%     dz_ind_plus,...
%     dz_ind_minus,...
%     3);

negative_id=L(1,2:n_sample+1)<0;
positive_id=L(1,2:n_sample+1)>0;

dL_sub=L(1,2:end);
negative_amp=zeros(1,n_sample);
positive_amp=zeros(1,n_sample);
negative_amp(negative_id)=dL_sub(1,negative_id);
positive_amp(positive_id)=dL_sub(1,positive_id);

% initial_H(1:end-2,end-1)=-negative_amp;
% initial_H(end-1,1:end-2)=-negative_amp;
% initial_H(1:end-2,end)=-positive_amp;
% initial_H(end,1:end-2)=-positive_amp;

ddL=[L(2:end,2:end) negative_amp' positive_amp';
     negative_amp 0 0
     positive_amp 0 0];

initial_H = -ddL;
initial_H(1:n_sample+2+1:end)=initial_H(1:n_sample+2+1:end)+...
    [y_init(1:end-1); y_init(end)/2-epsilon; y_init(end)/2+epsilon]';

rng(0);
fv_H=randn(n_sample+2,1);

disp('GDPA begins...');
initial_obj=sum(y_init);
disp(['GDPA obj: ' num2str(initial_obj)]);

tol_set=1e-5;
tol=Inf;
loop_i=0;
while tol>tol_set
    if loop_i==0
        [fv_H,...
            scaled_M,...
            scaled_factors] = ...
            compute_scalars_scal(...
            initial_H,...
            fv_H,...
            1); % compute scalars
        
        scaled_M_offdia=scaled_M;
        scaled_M_offdia(1:n_sample+2+1:end)=0;
        leftEnds=diag(initial_H)-sum(abs(scaled_M_offdia),2);
        leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));

    end

    %% define and solve GDPA Eq. (21) 
    
%     if mod(loop_i,2)==0
%         rho=0;
%     else
%         rho=-0.001;
%     end
    rho=0;
    [y,current_obj]=gdpa_core_tsp_test_uc( ...
        n_sample,...
        scaled_M,...
        ddL,...
        scaled_factors,...
        rho,...
        db,...
        options,...
        initial_H,...
        b_ind,...
        epsilon,...
        dz_ind_plus,...
        dz_ind_minus);

%     cb_f=0;
%     y=(cb_f*y_init+(1-cb_f)*y0)/1;
%     z=(cb_f*z_init+(1-cb_f)*z0)/1;
    
%     epsilon=sum(y);
%     epsilon=sum(abs(z));
%     epsilon=0;
%     [updated_H] = construct_H_tsp(n_sample,...
%         -L(2:end,2:end),...
%         epsilon,...
%         y,...
%         z,...
%         dz_ind_plus,...
%         dz_ind_minus,...
%         3);
%     
%     updated_H(1:end-2,end-1)=-negative_amp;
% updated_H(end-1,1:end-2)=-negative_amp;
% updated_H(1:end-2,end)=-positive_amp;
% updated_H(end,1:end-2)=-positive_amp;

updated_H = -ddL;
updated_H(1:n_sample+2+1:end)=updated_H(1:n_sample+2+1:end)+[y(1:end-1); y(end)/2-epsilon; y(end)/2+epsilon]';

% rng(0);
% fv_H=randn(n_sample+2,1);

    [fv_H,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars_scal(...
        updated_H,...
        fv_H,...
        1); % compute scalars
    
    scaled_M_offdia=scaled_M;
    scaled_M_offdia(1:n_sample+2+1:end)=0;
    leftEnds=diag(updated_H)-sum(abs(scaled_M_offdia),2);
    leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));


    initial_H=updated_H;

    disp(['GDPA obj ' num2str(current_obj)]);
    tol=norm(current_obj-initial_obj);
    initial_obj=current_obj;
    loop_i=loop_i+1;

%     y_init=y0;
%     z_init=z0;
end

disp('GDPA ends...');

%% the original_H converted from new_H
% original_H=[L zeros(n_sample,1);zeros(n_sample,1)' 0];
original_H=-L;
% original_H(1:n_sample+1+1:end)=[diag(L);0]+[y;-sum(z(dz_ind_minus))+0.5*u(n_sample+1)-epsilon];
original_H(1:n_sample+1+1:end)=original_H(1:n_sample+1+1:end)+[y(end) y(1:end-1)'];
% original_H(1:n_sample+1+1:end)=original_H(1:n_sample+1+1:end)+[y(1:n_sample); sum(y(n_sample+1)-sum(z(dz_ind_plus)))]';
% original_H(b_ind,n_sample+1)=0;
% original_H(n_sample+1,b_ind)=0;
original_H=(original_H+original_H')/2;

%% first eigenvector of the converted original_H

rng(0);
fv_H_0=randn(n_sample+1,1);

[fv_H1,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H,...
    1e-16,...
    1e3);

x_pred=sign(fv_H1);
% x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
x_val=x_pred(1)*x_pred(2:end);
% x_pred=sign(x_val);

obj_xlx=x_pred'*L*x_pred;
obj=sum(y);

% x_val(b_ind)=[];
% label(b_ind)=[];
err_count = sum(abs(sign(x_val) - label))/2;

figure; 
plot(x_val); 
xlabel('observation number'); 
ylabel('amplitude');
ylim([-2 2]);
xlim([1 n_sample]);
title('GDPA'); %Show the restored image
end