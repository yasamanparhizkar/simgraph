function [error_count,obj,t_orig_end] = glr_closed_form_uc(...
    L,...
    n_sample,...
    b_ind,...
    class_train_test)
%GLR_CLOSED_FORM Summary of this function goes here
%   Detailed explanation goes here
% % % % % % % initial_label_index=zeros(n_sample,1);
% % % % % % % initial_label_index(b_ind)=1;
% % % % % % % initial_label_index=logical(initial_label_index);
% % % % % % % t_orig=tic;
% % % % % % % x_pred=-pinv(L(~initial_label_index,~initial_label_index))...
% % % % % % %     *L(~initial_label_index,initial_label_index)...
% % % % % % %     *class_train_test(initial_label_index,:);
% % % % % % % t_orig_end=toc(t_orig);
% % % % % % % x_pred=sign(x_pred);
% % % % % % % x_valid=class_train_test;
% % % % % % % x_valid(~initial_label_index)=x_pred;
% % % % % % % class_train_test(b_ind)=[];
% % % % % % % error_count = sum(abs(sign(x_pred) - class_train_test))/2;
% % % % % % % x_valid=sign(x_valid);
% % % % % % % obj=x_valid'*L*x_valid;
t_orig=tic;
[v,d]=eig(full(L));
x_valid=sign(v(:,end));
obj=x_valid'*L*x_valid;
error_count=sum(abs(sign(x_valid(1)*x_valid(2:end)) - class_train_test))/2;
t_orig_end=toc(t_orig);

figure; 
plot(sign(x_valid(1)*x_valid(2:end))); 
xlabel('observation number'); 
ylabel('amplitude');
ylim([-2 2]);
xlim([1 n_sample]);
title('GLR'); %Show the restored image

% restored_s = reshape(x_valid(1)*x_valid(2:end)>0,39,51); figure; imshow(restored_s); title('GLR Spectral restored image'); %Show the restored image using the proposed BQP solver. 
% disp('Binary image (small) restoration problem');
% disp('Square image; Noise parameter: 0.1; Miu: 0.1; Image size: 39x51');
% disp(['GLR Spectral Objective function value:' num2str(obj)]);
% disp('.......................');
end

