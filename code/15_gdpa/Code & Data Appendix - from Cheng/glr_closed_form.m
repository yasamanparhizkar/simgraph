function [error_count,obj,t_orig_end] = glr_closed_form(...
    L,...
    n_sample,...
    b_ind,...
    class_train_test)
%GLR_CLOSED_FORM Summary of this function goes here
%   Detailed explanation goes here
initial_label_index=zeros(n_sample,1);
initial_label_index(b_ind)=1;
initial_label_index=logical(initial_label_index);
t_orig=tic;
x_pred=-pinv(L(~initial_label_index,~initial_label_index))...
    *L(~initial_label_index,initial_label_index)...
    *class_train_test(initial_label_index,:);
t_orig_end=toc(t_orig);
x_pred=sign(x_pred);
x_valid=class_train_test;
x_valid(~initial_label_index)=x_pred;
obj=x_valid'*L*x_valid;
class_train_test(b_ind)=[];
error_count = sum(abs(sign(x_pred) - class_train_test))/2;
% x_valid=sign(x_valid);
end

