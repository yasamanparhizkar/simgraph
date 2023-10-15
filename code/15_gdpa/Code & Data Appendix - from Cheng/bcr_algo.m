function [X,Y, t_orig_end] = bcr_algo(C,R,b,numiter,alpha,beta)
% [X, obj, obj_ph, t] = phasemax_imseg(A,B,b,1000,2,2,'G',m,5,false);
%  % phasemax parameters
%  beta = 2*norm(C,'fro');
%  gamma = beta/2; % anything between 0.3 and 0.5 works best
% clear all;load('temp4.mat');
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

m=size(C,1);
n=size(R,1);

%% make the signs of b's all non-negative
b_negative=b<0;
b(b_negative)=-b(b_negative);
R(b_negative,:)=-R(b_negative,:);

% initializing objective and time t
% obj = zeros(numiter,1);
% obj_ph = zeros(numiter,1);

% creating representation for phasemax

L0=cell(n,1);
for i=1:n
    L0{i}=sqrtm(reshape(R(i,:),[m m])); % matrix square root L
end

Ainv = (alpha*reshape(sum(R,1),[m m]) + C')^-1;

% wirtinger flow initialization
Yini=reshape(sum(b.*R,1),[m m])/n;
[V,D] = eigs(Yini,1);

% Initialization using "Gradient descent for rank minimization" algorithm
X0 = V*sqrt(abs(D)/2);

X = X0;

t_orig=tic;
tol=Inf;
iter=0;
while tol>1e-4
    iter=iter+1;
    % section 1: Compute optimal Q
    % solving for Q using LS
    Q=cell(n,1);
    LQ=0;
    Q_f_norm_sum=0;
    for i=1:n
        Q00=L0{i}*X;
        Q00_f=norm(Q00,'fro');
        Q0 = alpha*Q00_f/(alpha-beta);
        % projecting back Q onto frobenius ball
        Q{i} = min(Q0,sqrt(b(i))).*(Q00/Q00_f);
%         Q{i} = Q0.*(Q00/Q00_f); % modified that is different from the ECCV'16 paper!
        LQ=LQ+L0{i}'*Q{i};
        Q_f_norm_sum=Q_f_norm_sum+norm(Q{i},'fro');
    end
    % least square solver for X
    if iter==1
        initial_obj=X'*C*X-(beta/2)*Q_f_norm_sum;
    end
    X = Ainv*LQ;
    
    % objective function value
%     obj(iter) = X'*C*X;
    
%     obj_ph(iter) = obj(iter) - (beta/2)*Q_f_norm_sum;
    current_obj=X'*C*X-(beta/2)*Q_f_norm_sum;
    tol=norm(current_obj-initial_obj);
    initial_obj=current_obj;
    if iter>numiter
        break
    end
end
t_orig_end=toc(t_orig);
Y=X*X';
end