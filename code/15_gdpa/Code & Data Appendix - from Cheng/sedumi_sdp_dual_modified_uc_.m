function [x_pred,err_count,obj_xlx,obj,t_orig_end,dy,dz] = sedumi_sdp_dual_modified_uc_(dL,n_sample,b_ind,label,epsilon)
e_vec1=eye(n_sample+2);   %% identity matrice
for i=1:n_sample+2
    dA(:,:,i) = diag(e_vec1(:,i));
end  

% db = 2*label(b_ind); % training labels x 2

% dz_plus_idx = db < 0; % db<0, i.e., dz>0
% dz_minus_idx = db > 0; % db>0, i.e., dz<0
% dz_ind_plus = b_ind(dz_plus_idx);
% dz_ind_minus = b_ind(dz_minus_idx);

% dB_0=zeros(n_sample);
% e_vec0=zeros(n_sample,1);
% for i=1:length(dz_ind_minus)
%     e_vec2=zeros(n_sample,1);
%     e_vec2(dz_ind_minus(i))=1;
%     dB_nplus1(:,:,i)=[dB_0 e_vec2 e_vec0; e_vec2' 0 0; e_vec0' 0 0];
% end
% for i=1:length(dz_ind_plus)
%     e_vec2=zeros(n_sample,1);
%     e_vec2(dz_ind_plus(i))=1;
%     dB_nplus2(:,:,i)=[dB_0 e_vec0 e_vec2; e_vec0' 0 0; e_vec2' 0 0];
% end

dM = zeros(n_sample+2,n_sample+2); % number of total samples + 1

%ddL=[dL zeros(n_sample+1,1);zeros(1,n_sample+1) 0];

negative_id=dL(1,2:n_sample+1)<0;
positive_id=dL(1,2:n_sample+1)>0;

dL_sub=dL(1,2:end);
negative_amp=zeros(1,n_sample);
positive_amp=zeros(1,n_sample);
negative_amp(negative_id)=dL_sub(1,negative_id);
positive_amp(positive_id)=dL_sub(1,positive_id);

ddL=[dL(2:end,2:end) negative_amp' positive_amp';
     negative_amp 0 0
     positive_amp 0 0];

n_train=length(b_ind);

t_orig=tic;
cvx_begin sdp
variables dy(n_sample+1,1);
minimize(sum(dy))
subject to
counter_p=0;
counter_n=0;
for i=1:n_sample+2
    if i<=n_sample
        dM=dM+dy(i)*dA(:,:,i);
%         if i<=n_train
%             if db(i)>0 % db>0, i.e., dz<0
%                 counter_n=counter_n+1;
%                 dM=dM+dz(i)*dB_nplus1(:,:,counter_n);
%             else % db<0, i.e., dz>0
%                 counter_p=counter_p+1;
%                  dM=dM+dz(i)*dB_nplus2(:,:,counter_p);
%             end
%         end
    elseif i==n_sample+1
%         dM=dM+(((dy(n_sample+1)+sum(dz))*0.5)-epsilon-sum(dz(dz_ind_minus)))*dA(:,:,i);
          dM=dM+(((dy(n_sample+1)+0)*0.5)-epsilon-0)*dA(:,:,i);
%         dM=dM+dy(n_sample+1)*sw*dA(:,:,i);
    else % i==n_sample+2
%         dM=dM+(((dy(n_sample+1)+sum(dz))*0.5)+epsilon-sum(dz(dz_ind_plus)))*dA(:,:,i);
          dM=dM+(((dy(n_sample+1)+0)*0.5)+epsilon-0)*dA(:,:,i);
%         dM=dM+dy(n_sample+1)*(1-sw)*dA(:,:,i);
    end
end

dM-ddL>=0
cvx_end
t_orig_end=toc(t_orig);


%% convert back to the labels by computing the first eigenvector of the matrix H
% dL(n_sample+1,n_sample+1)=0;
original_H=-dL;

% original_H(1:n_sample+1+1:end)=diag(-dL)+[dy(1:n_sample); 2*dy(n_sample+1)+sum(dz(dz_ind_minus))-sum(dz(dz_ind_plus))];
original_H(1:n_sample+1+1:end)=original_H(1:n_sample+1+1:end)+[dy(end) dy(1:end-1)'];
% original_H(b_ind,end)=0;
% original_H(end,b_ind)=0;
dz=0;
H=(original_H+original_H')/2;

rng('default');
rng(0);
fv_H_0=randn(n_sample+1,1);

[fv_H,~] = ...
    lobpcg_fv(...
    fv_H_0,...
    H,...
    1e-16,...
    1e3);

x_pred=sign(fv_H);
% x_val = sign(label(b_ind(1))*sign(fv_H(1)))*sign(fv_H(1:n_sample));
x_val=x_pred(1)*x_pred(2:end);
% x_pred=sign(x_val);

% obj=x_pred'*-dL(1:end-1,1:end-1)*x_pred;
obj_xlx=x_pred'*dL*x_pred;
obj=sum(dy);

% x_val(b_ind)=[];
% label(b_ind)=[];

err_count = sum(abs(sign(x_val) - label))/2;

figure; 
plot(x_val); 
xlabel('observation number'); 
ylabel('amplitude');
ylim([-2 2]);
xlim([1 n_sample]);
title('SDP dual modified'); %Show the restored image
end

