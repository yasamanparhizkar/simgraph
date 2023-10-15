function [error_count,obj_,obj_sns,t_orig_end] = sns_uc(L,n_sample,b_ind,class_train_test)
%SNS Summary of this function goes here
%   Detailed explanation goes here
known_size=length(b_ind);
A=[eye(known_size) zeros(known_size,n_sample-known_size)]';
b=class_train_test(b_ind)';
t_orig=tic;
[predicted_label,obj_sns,~]=Fast_BQP_Solver(L,[],[],[],[]);
t_orig_end=toc(t_orig);
predicted_label_=predicted_label(1)*predicted_label(2:end);
obj_=predicted_label'*L*predicted_label;
% predicted_label(b_ind)=[];
% class_train_test(b_ind)=[];
error_count=sum(abs(predicted_label(1)*predicted_label(2:end)-class_train_test))/2;
% error_count=min([sum(abs(predicted_label-class_train_test))/2 sum(abs(-predicted_label-class_train_test))/2]);
% error_count=0;

figure; 
plot(predicted_label_); 
xlabel('observation number'); 
ylabel('amplitude');
ylim([-2 2]);
xlim([1 n_sample]);
title('SNS'); %Show the restored image

% restored_s = reshape(predicted_label(1)*predicted_label(2:end)>0,39,51); figure; imshow(restored_s); title('SNS restored image'); %Show the restored image using the proposed BQP solver. 
% disp('Binary image (small) restoration problem');
% disp('Square image; Noise parameter: 0.1; Miu: 0.1; Image size: 39x51');
% disp(['SNS Objective function value:' num2str(obj)]);
% disp('.......................');
end

