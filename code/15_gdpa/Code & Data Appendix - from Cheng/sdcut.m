function [dy,dz,obj,db,x_pred,err_count,t_orig_end,u_vec,alpha] = sdcut(label,b_ind,n_sample,dA,dB,dL)
db = 2*label(b_ind); % training labels x 2

%% this is for CDCS ADMM SDP==========================================
% % % % % % % opts.maxIter = 1e+3;
% % % % % % % opts.relTol  = 1e-3;
% % % % % % % opts.solver = 'dual';
A=zeros(n_sample+1+length(b_ind),(n_sample+1)^2);
for i=1:n_sample+1
    A(i,:)=vec(dA(:,:,i))';
end
for i=1:length(b_ind)
    A(n_sample+1+i,:)=vec(dB(:,:,i))';
end
b=zeros(n_sample+1+length(b_ind),1);
b(1:n_sample+1)=1;
b(n_sample+1+1:n_sample+1+length(b_ind))=db;
% % % % % % % c=vec(-dL)';
% % % % % % % At=A';
% % % % % % % K.s = n_sample+1; % PSD matrix dimension
% % % % % % % [x_cdcs,y_cdcs,z_cdcs,info_cdcs] = cdcs(At,b,c,K,opts); % SDP via an ADMM approach.
% % % % % % % 
% % % % % % % dy=y_cdcs(1:n_sample+1);
% % % % % % % dz=y_cdcs(n_sample+1+1:n_sample+1+length(b_ind));
% % % % % % % obj=info_cdcs.cost;
% % % % % % % t_orig_end=info_cdcs.time.admm;
% % % % % % % H=reshape(z_cdcs,[n_sample+1 n_sample+1]);
%% this is for CDCS ADMM SDP ends=====================================

%% this is for SDCUT SDP==============================================
% get data and options
data.A=dL; % mat(n_sample+1,n_sample+1)
B=cell(1,n_sample+1+length(b_ind));
for i=1:n_sample+1+length(b_ind)
    B{i}=reshape(A(i,:),[n_sample+1 n_sample+1]);
end
data.B=B; % cell(1,n_sample+1+length(b_ind))
data.b=b; % vec(n_sample+1+length(b_ind),1)
data.u_init=zeros(n_sample+1+length(b_ind),1);

dz_plus_idx = db < 0;
dz_minus_idx = db > 0;
dz_ind_plus = b_ind(dz_plus_idx);
dz_ind_minus = b_ind(dz_minus_idx);

data.u_init(b_ind)=1;
data.u_init(n_sample+1)=length(b_ind);
data.u_init(n_sample+1+dz_ind_minus)=-1;
data.u_init(n_sample+1+dz_ind_plus)=1;

data.l_bbox=zeros(n_sample+1+length(b_ind),1)-Inf; 
data.u_bbox=zeros(n_sample+1+length(b_ind),1)+Inf;

% % % dz_plus_idx = db < 0;
% % % dz_minus_idx = db > 0;
% % % dz_ind_plus = b_ind(dz_plus_idx);
% % % dz_ind_minus = b_ind(dz_minus_idx);
% % % data.l_bbox(n_sample+1+dz_ind_plus)=0;
% % % data.u_bbox(n_sample+1+dz_ind_minus)=0;

opts.sigma=1e2; % between 1e-4 and 1e-2 according to the cvpr'13 paper
opts.lbfgsb_maxIts=1e2;
opts.lbfgsb_factr=1e7;
opts.lbfgsb_pgtol=1e-3;
opts.lbfgsb_m=2e2;
opts.lbfgsb_printEvery=1e2;
% solve dual
[u_opt,C_minus_opt,results]=sdcut_lbfgsb(data,opts);
% get optimal lifted primal X=x*x'
X_opt = (-0.5 / opts.sigma) * C_minus_opt;
% recover x from X=x*x'
dy=u_opt(1:n_sample+1);
dz=u_opt(n_sample+1+1:n_sample+1+length(b_ind));
obj=results.obj;
t_orig_end=results.time;
disp(['sdcut lbfgsb iter: ' num2str(results.iters)]);
H=[-dL(1:n_sample,1:n_sample)+diag(dy(1:n_sample)) [dz;zeros(n_sample-length(b_ind),1)];...
    [dz;zeros(n_sample-length(b_ind),1)]' dy(n_sample+1)];
%% this is for SDCUT SDP ends=========================================

rng('default');
rng(0);
fv_H_0=randn(n_sample+1,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    H,...
    1e-16,...
    1e3);

x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
x_pred=sign(x_val);
x_val(b_ind)=[];
label(b_ind)=[];
err_count = sum(abs(sign(x_val) - label))/2;
x_val2 = sign(X_opt(1:n_sample,n_sample+1));
% x_pred=sign(x_val);
x_val2(b_ind)=[];
err_count2 = sum(abs(sign(x_val2) - label))/2;
disp(['error_count dual: ' num2str(err_count) ' | error_count primal: ' num2str(err_count2)]);

H_offdia=H;
H_offdia(1:n_sample+1+1:end)=0;

u_vec=diag(H)+sum(H_offdia,2);

db_plus_idx = db > 0;
db_minus_idx = db < 0;
db_plus_idx = b_ind(db_plus_idx);
db_minus_idx = b_ind(db_minus_idx);

w_positive=H(db_plus_idx,end); % corres. to db_plus_idx
w_positive=-w_positive;
w_negative=H(db_minus_idx,end); % corres. to db_minus_idx
w_negative=-w_negative;
delta_xNplus1_xi_positive=fv_H(end)-fv_H(db_plus_idx);
delta_xNplus1_xi_negative=fv_H(end)-fv_H(db_minus_idx);
sNplus1=sum(w_positive.*delta_xNplus1_xi_positive);
sNplus2=sum(w_negative.*delta_xNplus1_xi_negative);
alpha=(sNplus1-sNplus2)/(2*fv_H(end));
end

