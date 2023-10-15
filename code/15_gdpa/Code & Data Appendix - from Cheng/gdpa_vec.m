function [current_obj,x_pred,err_count,u_vec,alpha,eigen_gap,t_orig_end] = ...
    gdpa_vec(label,b_ind,n_sample,cL,u,alpha,sw,...
    rho)

%% LP settings
options=optimoptions('linprog','Algorithm','interior-point','display','none'); % linear program (LP) setting for Frank-Wolfe algorithm
% options=optimoptions('linprog','Algorithm','interior-point'); % linear program (LP) setting for Frank-Wolfe algorithm
options.OptimalityTolerance=1e-5; % LP optimality tolerance
options.ConstraintTolerance=1e-5; % LP interior-point constraint tolerance

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;
dz_ind_plus = b_ind(dz_plus_idx);
dz_ind_minus = b_ind(dz_minus_idx);

%% initialize a psd matrix ABL
scalee=1/alpha;

sccc=1;
y0=zeros(n_sample,1);
y0(b_ind)=1*sccc;
dz_LP_test_init=-(db/2)*sccc;
ynplus1=sum(y0)*1e0;
dy_LP_test_init=[y0;ynplus1];
alpha=sw*(ynplus1+sum(dz_LP_test_init));

cL=cL*scalee;

% y0=zeros(n_sample,1);
% y0(b_ind)=1;
% dy_LP_test_init=[y0;sum(y0(dz_ind_minus));sum(y0(dz_ind_plus))];
% dz_LP_test_init=-db/2;

% mplus=sum(dz_LP_test_init(dz_ind_plus));
% mminus=-sum(dz_LP_test_init(dz_ind_minus));
% mmm=mminus/mplus;
% 
% alpha=(1/(1+mmm))*(...
%        (ynplus1+sum(dz_LP_test_init))*(sw+(sw-1)*mmm)...
%        -sum(dz_LP_test_init(dz_ind_minus))...
%        +mmm*sum(dz_LP_test_init(dz_ind_plus))...
%        );
   
% scalee=1;

[initial_H] = construct_H(sw,n_sample,...
    cL,...
    u,...
    alpha,...
    dy_LP_test_init,...
    dz_LP_test_init,...
    dz_ind_plus,...
    dz_ind_minus,...
    3);

% initial_H = initial_H+-min(eig(initial_H))*eye(n_sample+2);

rng(0);
fv_H=randn(n_sample+2,1);
[fv_H,~] = ...
    lobpcg_fv(...
    fv_H,...
    initial_H*1/scalee,...
    1e-4,...
    200);
disp(['energy in b_ind n_sample+1 n_sample+2: ' num2str(norm(fv_H([b_ind n_sample+1 n_sample+2])))...
            ' | energy in z: ' num2str(norm(fv_H(length(b_ind)+1:end-2)))]);

%% check error before LP==============================
y=dy_LP_test_init;
z=dz_LP_test_init;
original_H=[cL zeros(n_sample,1);zeros(n_sample,1)' 0];
original_H(1:n_sample+1+1:end)=[diag(cL);0]+y;
original_H(b_ind,n_sample+1)=z;
original_H(n_sample+1,b_ind)=z;
original_H=(original_H+original_H')/2;
rng(0);
fv_H_0=randn(n_sample+1,1);
% t_fv1=tic;
[fv_H1,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H,...
    1e-4,...
    200);
% t_1=toc(t_fv1);
ene1=norm(fv_H1(length(b_ind)+1:end-1));
[fv_H2,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H*1/scalee,...
    1e-4,...
    200);
ene2=norm(fv_H2(length(b_ind)+1:end-1));
if ene1<ene2
   fv_H=fv_H1; 
else
    fv_H=fv_H2;
end
disp(['energy in b_ind n_sample+1: ' num2str(norm(fv_H([b_ind n_sample+1])))...
    ' | energy in z: ' num2str(norm(fv_H(length(b_ind)+1:end-1)))]);
x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
err_count0 = sum(abs(sign(x_val) - label))/2;
disp(['error_count before LP: ' num2str(err_count0)]);
%%====================================================

% disp(['mineig initial_H: ' num2str(min(eig(initial_H)))]);

rng(0);
fv_H=randn(n_sample+2,1);

initial_obj=sum(dy_LP_test_init)+db'*dz_LP_test_init;
% disp(['v3 LP main iteration ' num2str(0) ' | current obj: ' num2str(initial_obj) ' | mineig: ' num2str(min(eig(initial_H)))]);

t_orig=tic;

tol_set=1e-4;
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
            scalee); % compute scalars
        
%         scaled_M_offdia=scaled_M;
%         scaled_M_offdia(1:n_sample+2+1:end)=0;
%         leftEnds=diag(initial_H)-sum(abs(scaled_M_offdia),2);
%         leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));
%         disp(['energy in b_ind n_sample+1 n_sample+2: ' num2str(norm(fv_H([b_ind n_sample+1 n_sample+2])))...
%             ' | energy in z: ' num2str(norm(fv_H(length(b_ind)+1:end-2)))]);
%         disp(['v3 LP before LP LeftEnds mean: ' num2str(mean(leftEnds)) ' | LeftEnds difference: ' num2str(leftEnds_diff)]);
    end

    [y0,z0,current_obj]=gdpa_core_vec( ...
        n_sample,...
        scaled_M,...
        scaled_factors,...
        rho,...
        db,...
        options,...
        cL,...
        b_ind,...
        u,...
        alpha,...
        sw,...
        dz_ind_plus,...
        dz_ind_minus);
    

% mplus=sum(z0(dz_ind_plus));
% mminus=-sum(z0(dz_ind_minus));
% mmm=mminus/mplus;
% 
% alpha=(1/(1+mmm))*(...
%        (y0(n_sample+1)+sum(z0))*(sw+(sw-1)*mmm)...
%        -sum(z0(dz_ind_minus))...
%        +mmm*sum(z0(dz_ind_plus))...
%        );
alpha=sw*(y0(n_sample+1)+sum(z0));    
    y=y0;
    z=z0;
    [updated_H] = construct_H(sw,n_sample,...
        cL,...
        u,...
        alpha,...
        y,...
        z,...
        dz_ind_plus,...
        dz_ind_minus,...
        3);
    
    % %% the solution new_H
    % updated_H=(updated_H+updated_H')/2;
    %
    % %% the original_H converted from new_H
    % original_H=[cL zeros(n_sample,1);zeros(n_sample,1)' 0];
    % % original_H(1:n_sample+1+1:end)=[diag(cL);0]+[y;-sum(z(dz_ind_minus))+0.5*u(n_sample+1)-alpha];
    % original_H(1:n_sample+1+1:end)=[diag(cL);0]+y;
    % % original_H(1:n_sample+1+1:end)=original_H(1:n_sample+1+1:end)+[y(1:n_sample); sum(y(n_sample+1)-sum(z(dz_ind_plus)))]';
    % original_H(b_ind,n_sample+1)=z;
    % original_H(n_sample+1,b_ind)=z;
    %
    % original_H=(original_H+original_H')/2;
    %
    % rng(0);
    % fv_Ho_0=randn(n_sample+1,1);
    % [alpha] = compute_alpha(...
    %     n_sample,...
    %     db,...
    %     b_ind,...
    %     original_H,...
    %     fv_Ho_0);
    
%     rng(0);
%     fv_H=randn(n_sample+2,1);
    
    [fv_H,...
        scaled_M,...
        scaled_factors] = ...
        compute_scalars_scal(...
        updated_H,...
        fv_H,...
        scalee); % compute scalars
    
%     disp(['energy in b_ind n_sample+1 n_sample+2: ' num2str(norm(fv_H([b_ind n_sample+1 n_sample+2])))...
%         ' | energy in z: ' num2str(norm(fv_H(length(b_ind)+1:end-2)))]);
%     
%     scaled_M_offdia=scaled_M;
%     scaled_M_offdia(1:n_sample+2+1:end)=0;
%     leftEnds=diag(updated_H)-sum(abs(scaled_M_offdia),2);
%     leftEnds_diff=sum(abs(leftEnds-mean(leftEnds)));
%     disp(['v3 LP after LP LeftEnds mean: ' num2str(mean(leftEnds)) ' | LeftEnds difference: ' num2str(leftEnds_diff)]);
    
    disp(['v3 LP obj ' num2str(current_obj)]);
    tol=norm(current_obj-initial_obj);
    initial_obj=current_obj;
    loop_i=loop_i+1;
end

t_2=toc(t_orig);

%% the solution new_H
updated_H=(updated_H+updated_H')/2;

%% the original_H converted from new_H
original_H=[cL zeros(n_sample,1);zeros(n_sample,1)' 0];
% original_H(1:n_sample+1+1:end)=[diag(cL);0]+[y;-sum(z(dz_ind_minus))+0.5*u(n_sample+1)-alpha];
original_H(1:n_sample+1+1:end)=[diag(cL);0]+y;
% original_H(1:n_sample+1+1:end)=original_H(1:n_sample+1+1:end)+[y(1:n_sample); sum(y(n_sample+1)-sum(z(dz_ind_plus)))]';
original_H(b_ind,n_sample+1)=z;
original_H(n_sample+1,b_ind)=z;

original_H=(original_H+original_H')/2;

H_offdia=original_H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(original_H)+sum(H_offdia,2);

%% eigen-gap
eigen_gap=0;
% eigen_gap=min(eig(updated_H))-min(eig(original_H));

% test_v=-1e4-200;
% rr=zeros(100,2);
% for ii=1:100
%     test_v=test_v+200;
%     original_H(end,end)=test_v;
%% first eigenvector of the converted original_H

rng(0);
fv_H_0=randn(n_sample+1,1);
% t_fv2=tic;
[fv_H1,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H,...
    1e-4,...
    200);
% t_3=toc(t_fv2);
t_orig_end=t_2;
ene1=norm(fv_H1(length(b_ind)+1:end-1));
[fv_H2,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H*1/scalee,...
    1e-4,...
    200);
ene2=norm(fv_H2(length(b_ind)+1:end-1));
if ene1<ene2
   fv_H=fv_H1; 
else
    fv_H=fv_H2;
end
disp(['energy in b_ind n_sample+1: ' num2str(norm(fv_H([b_ind n_sample+1])))...
    ' | energy in z: ' num2str(norm(fv_H(length(b_ind)+1:end-1)))]);

% [v,~]=eig(original_H*1/scalee);
% fv_H=v(:,1);

% [alpha] = compute_alpha(...
%     n_sample,...
%     db,...
%     b_ind,...
%     original_H,...
%     fv_H);

x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% x_val = sign(fv_H(1:n_sample));
x_pred=sign(x_val);
err_count = sum(abs(sign(x_val) - label))/2;
% disp(['cond H: ' num2str(cond(original_H)) ' | cond H bar: ' num2str(cond(updated_H))]);
% rr(ii,1)=test_v;
% rr(ii,2)=err_count;
%
% end
%
% figure();plot(rr(:,1),rr(:,2));
end