function [obj,err_count,t_orig_end] = cdcs20(label,b_ind,n_sample,dA,dB,dL,u,alpha,sw)

db = 2*label(b_ind); % training labels x 2
dz_plus_idx = db < 0;
dz_minus_idx = db > 0;
dz_ind_plus = b_ind(dz_plus_idx);
dz_ind_minus = b_ind(dz_minus_idx);

opts.maxIter = 1e+3;
opts.relTol  = 1e-3;
opts.solver = 'dual';
opts.verbose = 0;
A=zeros(n_sample+2+length(b_ind),(n_sample+2)^2);
for i=1:n_sample
    A(i,:)=vec([dA(:,:,i) zeros(n_sample+1,1); zeros(1,n_sample+1) 0])';
end

Bnp1 = zeros(n_sample+2);
Bnp2 = zeros(n_sample+2);
for i=1:length(dz_ind_minus)
    Bnp1 = Bnp1+[dB(:,:,dz_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    %dz(dz_ind_minus(i))<0;
end
for i=1:length(dz_ind_plus)
    dBnp2_i=dB(:,:,dz_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = Bnp2+[zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    %dz(dz_ind_plus(i))>0;
end
dA_nplus1=[dA(:,:,n_sample+1) zeros(n_sample+1,1); zeros(1,n_sample+1) 0];
A(n_sample+1,:)=vec(sw*(dA_nplus1+Bnp1+Bnp2)-Bnp1)';
dA_nplus2=zeros(n_sample+2);
dA_nplus2(n_sample+2,n_sample+2)=1;
A(n_sample+2,:)=vec((1-sw)*(dA_nplus2+Bnp1+Bnp2)-Bnp2)';

for i=1:length(dz_ind_minus)
    Bnp1 = [dB(:,:,dz_ind_minus(i)) zeros(n_sample+1,1);zeros(n_sample+1,1)' 0];
    %dz(dz_ind_minus(i))<0;
    A(n_sample+2+dz_ind_minus(i),:)=vec(Bnp1)';
end
for i=1:length(dz_ind_plus)
    dBnp2_i=dB(:,:,dz_ind_plus(i));
    dBnp2_i=dBnp2_i(:,end);
    Bnp2 = [zeros(n_sample+1) dBnp2_i;dBnp2_i' 0];
    %dz(dz_ind_plus(i))>0;
    A(n_sample+2+dz_ind_plus(i),:)=vec(Bnp2)';
end

b=zeros(n_sample+2+length(b_ind),1);
b(1:n_sample)=1;

b(n_sample+1)=sw*(1+sum(db))-sum(db(dz_ind_minus)); %?
b(n_sample+2)=(1-sw)*(1+sum(db))-sum(db(dz_ind_plus)); %?

b(n_sample+2+1:n_sample+2+length(b_ind))=db;

dL(n_sample+1,n_sample+1)=alpha;
c=-vec([-dL zeros(n_sample+1,1);zeros(n_sample+1,1)' alpha])';

At=A';
K.s = n_sample+2; % PSD matrix dimension
[x_cdcs,y_cdcs,z_cdcs,info_cdcs] = cdcs(At,b,c,K,opts); % SDP via an ADMM approach.

H=reshape(z_cdcs,[n_sample+2 n_sample+2]);
y_cdcs=-y_cdcs;
dy=[y_cdcs(1:n_sample); y_cdcs(n_sample+1)+y_cdcs(n_sample+2)];
dz=y_cdcs(n_sample+2+1:n_sample+2+length(b_ind));
obj=info_cdcs.cost;
t_orig_end=info_cdcs.time.admm;

%% the original_H converted from new_H
dL(n_sample+1,n_sample+1)=0;
original_H=-dL;

% original_H(1:n_sample+1+1:end)=diag(-dL)+[dy(1:n_sample); 2*dy(n_sample+1)+sum(dz(dz_ind_minus))-sum(dz(dz_ind_plus))];
original_H(1:n_sample+1+1:end)=diag(-dL)+dy;
original_H(b_ind,end)=dz;
original_H(end,b_ind)=dz;
original_H=(original_H+original_H')/2;

rng('default');
rng(0);
fv_H_0=randn(n_sample+1,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    original_H,...
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

%% eigen-gap
eigen_gap=min(eig(H))-min(eig(original_H));

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
%% =======================
end