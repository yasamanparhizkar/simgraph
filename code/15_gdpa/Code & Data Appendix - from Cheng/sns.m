function [error_count,obj,t_orig_end] = sns(L,n_sample,b_ind,class_train_test)
%SNS Summary of this function goes here
%   Detailed explanation goes here
known_size=length(b_ind);
A=[eye(known_size) zeros(known_size,n_sample-known_size)]';
b=class_train_test(b_ind)';
t_orig=tic;
[predicted_label,obj,~]=Fast_BQP_Solver(L,A,b,[],[]);
t_orig_end=toc(t_orig);
predicted_label(b_ind)=[];
class_train_test(b_ind)=[];
error_count=min([sum(abs(predicted_label-class_train_test))/2 sum(abs(-predicted_label-class_train_test))/2]);
end

