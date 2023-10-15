function [x_pred,err_count,obj_xlx,obj,t_orig_end] = sedumi_sdp_dual_uc(dL,dA,n_sample,b_ind,label)
db = 2*label(b_ind); % training labels x 2
dM = zeros(n_sample+1,n_sample+1); % number of total samples + 1

t_orig=tic;
cvx_begin sdp
variables dy(n_sample+1,1);
minimize(sum(dy))
subject to
for i=1:n_sample+1
    dM = dM + dy(i)*dA(:,:,i);
end
% for i=1:length(b_ind)
%     dM = dM + dz(i)*dB(:,:,i);
% end
dM-dL>=0
cvx_end
t_orig_end=toc(t_orig);

H=dM-dL;
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

% obj=x_pred'*dL(1:end-1,1:end-1)*x_pred;
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
title('SDP dual'); %Show the restored image
end

