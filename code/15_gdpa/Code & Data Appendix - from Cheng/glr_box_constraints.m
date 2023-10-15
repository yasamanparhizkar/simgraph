function [error_count,obj,t_orig_end] = glr_box_constraints(cL,n_sample,b_ind,label)


t_orig=tic;
cvx_begin 
cvx_solver SeDuMi
variables x(n_sample,1);
minimize(x'*cL*x)
subject to
x(b_ind)==label(b_ind);
for i=length(b_ind)+1:n_sample
    -1<=x(i)<=1;
end
cvx_end
t_orig_end=toc(t_orig);
% t_orig_end
obj=sign(x)'*cL*sign(x);
x(b_ind)=[];
label(b_ind)=[];
error_count = sum(abs(sign(x) - label))/2;
end