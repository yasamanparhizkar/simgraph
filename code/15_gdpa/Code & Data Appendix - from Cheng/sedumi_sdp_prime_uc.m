function [x_pred,err_count,obj_xlx,obj_sdp_primal,t_orig_end] = sedumi_sdp_prime_uc(dL,n_sample,b_ind,label)
label_b_ind=label(b_ind);

t_orig=tic;
cvx_begin sdp
cvx_solver SeDuMi
variables M(n_sample+1,n_sample+1);
maximize(sum(sum(dL.*M)))
subject to
M(1:n_sample+1+1:end)==1;
% for i=1:length(b_ind)
%     sum(sum((dB(:,:,i).*M)))==2*label_b_ind(i);
% end
M>=0
cvx_end
t_orig_end=toc(t_orig);

% x_val=M(1:end-1,end);
x_val=M(:,end);
x_pred=sign(x_val);

% obj_xlx=x_pred'*dL(1:end-1,1:end-1)*x_pred;
obj_xlx=x_pred'*dL*x_pred;
obj_sdp_primal=sum(sum(dL.*M));

% x_val(b_ind)=[];
% label(b_ind)=[];
err_count = sum(abs(sign(x_val(1)*x_val(2:end)) - label))/2;

figure; 
plot(sign(x_val(1)*x_val(2:end))); 
xlabel('observation number'); 
ylabel('amplitude');
ylim([-2 2]);
xlim([1 n_sample]);
title('SDP primal'); %Show the restored image

end