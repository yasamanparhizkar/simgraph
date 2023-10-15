function [err_count,t_orig_end] = bcr(label,b_ind,n_sample,dA,dB,dL)
db = 2*label(b_ind); % training labels x 2

% opts.maxIter = 1e+4;
% opts.relTol  = 1e-3;
% opts.solver = 'primal';
% opts.verbose = 0;
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
% c=vec(-dL)';
% At=A';
% K.s = n_sample+1; % PSD matrix dimension
% [x_cdcs,y_cdcs,z_cdcs,info_cdcs] = cdcs(At,b,c,K,opts); % SDP via an ADMM approach.

C=dL;
R=A;
numiter=1e2;
beta=norm(dL,'fro');
alpha=2*beta;
[X,~,t_orig_end] = bcr_algo(C,R,b,numiter,alpha,beta);
% [X, obj, obj_ph, t] = phasemax_imseg(A,B,b,1000,2,2,'G',m,5,false);
%  % phasemax parameters
%  beta = 2*norm(C,'fro');
%  gamma = beta/2; % anything between 0.3 and 0.5 works best
% Input
% C     : matrix used in objective for image seg x'Cx
% R     : vector or matrix consisting every constraints. Each row vector
%       correspond to single constraint in the formulation
% b     : vector of value bounding each constraint
% beta  : weight parameter, beta > 1
% gamma : weighting between objective function and phasemax function
% alpha : variable for x^Te = alpha
% method: whether the constraints are greater than or less than, 'G','L'
% N     : Number of vertices
% numiter: number of iteration
% Output
% X     : Solution in real number
% obj   : objective value of image segmentation problem
% obj_ph: objective value of phasemax form of problem. This should decrease
% t     : recorded time of every iteration

% dy=y_cdcs(1:n_sample+1);
% dz=y_cdcs(n_sample+1+1:n_sample+1+length(b_ind));

% obj=info_cdcs.cost;
% t_orig_end=info_cdcs.time.admm;

% H=reshape(z_cdcs,[n_sample+1 n_sample+1]);

% rng('default');
% rng(0);
% fv_H_0=randn(n_sample+1,1);
% 
% [fv_H,~] = ...
%     lobpcg_fv(...
%     fv_H_0,...
%     Y,...
%     1e-16,...
%     1e3);
% 
% x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
% % x_val = sign(fv_H(1:n_sample));
% x_pred=sign(x_val);
% err_count = sum(abs(sign(x_val) - label))/2;

known=X(b_ind);
unknown=X(length(b_ind)+1:end-1);
label(b_ind)=[];
% err_count1=sum(abs(sign([known;unknown]) - label))/2;
% err_count2=sum(abs(sign([-known;unknown]) - label))/2;
% err_count3=sum(abs(sign([known;-unknown]) - label))/2;
% err_count4=sum(abs(sign([-known;-unknown]) - label))/2;
err_count1=sum(abs(sign(unknown) - label))/2;
err_count2=sum(abs(sign(-unknown) - label))/2;

% err_count = min([err_count1 err_count2 err_count3 err_count4]);
err_count = min([err_count1 err_count2]);

% H_offdia=H;
% H_offdia(1:n_sample+1+1:end)=0;
% 
% u_vec=diag(H)+sum(H_offdia,2);
% 
% db_plus_idx = db > 0;
% db_minus_idx = db < 0;
% db_plus_idx = b_ind(db_plus_idx);
% db_minus_idx = b_ind(db_minus_idx);
% 
% w_positive=H(db_plus_idx,end); % corres. to db_plus_idx
% w_positive=-w_positive;
% w_negative=H(db_minus_idx,end); % corres. to db_minus_idx
% w_negative=-w_negative;
% delta_xNplus1_xi_positive=fv_H(end)-fv_H(db_plus_idx);
% delta_xNplus1_xi_negative=fv_H(end)-fv_H(db_minus_idx);
% sNplus1=sum(w_positive.*delta_xNplus1_xi_positive);
% sNplus2=sum(w_negative.*delta_xNplus1_xi_negative);
% alpha=(sNplus1-sNplus2)/(2*fv_H(end));
end

