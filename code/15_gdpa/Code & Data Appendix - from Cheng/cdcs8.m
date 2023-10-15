function [dy,dz,obj,db,x_pred,err_count,t_orig_end,u_vec,alpha] = cdcs8(label,b_ind,n_sample,dA,dB,dL)
db = 2*label(b_ind); % training labels x 2

opts.maxIter = 1e+3;
opts.relTol  = 1e-3;
opts.solver = 'primal';
opts.verbose = 0;
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
c=vec(-dL)';
At=A';
K.s = n_sample+1; % PSD matrix dimension
[x_cdcs,y_cdcs,z_cdcs,info_cdcs] = cdcs(At,b,c,K,opts); % SDP via an ADMM approach.

dy=y_cdcs(1:n_sample+1);
dz=y_cdcs(n_sample+1+1:n_sample+1+length(b_ind));

obj=info_cdcs.cost;
t_orig_end=info_cdcs.time.admm;

H=reshape(z_cdcs,[n_sample+1 n_sample+1]);

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
% x_val = sign(fv_H(1:n_sample));
x_pred=sign(x_val);
x_val(b_ind)=[];
label(b_ind)=[];
err_count = sum(abs(sign(x_val) - label))/2;

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

