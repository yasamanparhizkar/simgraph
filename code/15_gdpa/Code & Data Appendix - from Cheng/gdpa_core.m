 function [y,z,obj]=gdpa_core( ...
    n,...
    scaled_M,...
    scaled_factors,...
    rho,...
    db,...
    options,...
    cL,...
    b_ind,...
    u,...
    alpha,...
    sw,...
    dz_ind_plus,...
    dz_ind_minus)

lbi=length(b_ind); % number of known labels

scaled_M_n=scaled_M(1:n,1:n);
scaled_M_n(1:n+1:end)=0;

%% define [c] [P] [d] in Eq. \eqref{eq:LP_std1} in paper
c=[ones(n+1,1); db(:); zeros(lbi,1)]; % N+1+2*M
d=zeros(n+1+2*lbi,1); % N+2+M
d(1:n)=diag(cL)-sum(abs(scaled_M_n),2);
%%========================
d(1:n+1)=d(1:n+1)-rho; % ensure PD
%%========================

%% ======define P starts=======
P=zeros(n+1+2*lbi,n+1+2*lbi);
% the first n rows
P(1:n,1:n)=-eye(n);

for lbi_i=1:lbi
    if sign(db(lbi_i))==1 % b_i>0, i.e., z_i<0
        scalars=abs(scaled_factors(lbi_i,n+1));   
        % the n+1 th row
        P(n+1,n+1+lbi+lbi_i)=abs(scaled_factors(n+1,lbi_i));  % X'_{1xM}
    else                  % b_i<0, i.e., z_i>0
        scalars=abs(scaled_factors(lbi_i,n+2));
        % the n+1 th row
        P(n+1,n+1+lbi+lbi_i)=abs(scaled_factors(n+2,lbi_i)); % Y'_{1xM}
    end
    P(lbi_i,n+1+lbi+lbi_i)=scalars; % E_{NxM}
end
% the n+1 th row
P(n+1,n+1)=-1;
% the next lbi rows
P(n+1+1:n+1+lbi,n+1+1:n+1+lbi)=eye(lbi);
P(n+1+1:n+1+lbi,n+1+lbi+1:n+1+2*lbi)=-eye(lbi);
% the last lbi rows
P(n+1+lbi+1:n+1+2*lbi,n+1+1:n+1+lbi)=-eye(lbi);
P(n+1+lbi+1:n+1+2*lbi,n+1+lbi+1:n+1+2*lbi)=-eye(lbi);
%% ======define P ends=======

% Aeq=zeros(lbi-2,n+1+2*lbi);
% 
% % Aeq(lbi-1,n+1+dz_ind_minus(1))=1;
% % Aeq(lbi-1,n+1+dz_ind_plus(1))=1;
% 
% beq=zeros(lbi-2,1);
% 
% for i=1:length(dz_ind_minus)-1
%     Aeq(i,n+1+dz_ind_minus(1))=1;
%     Aeq(i,n+1+dz_ind_minus(1+i))=-1;
% end
% for i=1:length(dz_ind_plus)-1
%     Aeq(length(dz_ind_minus)-1+i,n+1+dz_ind_plus(1))=1;
%     Aeq(length(dz_ind_minus)-1+i,n+1+dz_ind_plus(1+i))=-1;
% end

% lb=zeros(n+1+2*lbi,1);
% ub=zeros(n+1+2*lbi,1);
% lb(1:n+1)=-Inf;
% ub(1:n+1)=Inf;
% lb(n+1+dz_ind_minus)=-Inf;
% ub(n+1+dz_ind_minus)=0;
% lb(n+1+dz_ind_plus)=0;
% ub(n+1+dz_ind_plus)=Inf;
% lb(n+1+lbi+1:n+1+2*lbi)=0;
% ub(n+1+lbi+1:n+1+2*lbi)=Inf;

[s_k,obj] = linprog(c,...
    P,d,...
    [],[],...
    [],[],options);

s_k=s_k(1:n+1+lbi);
y=s_k(1:n+1);
z=s_k(n+1+1:n+1+lbi);
end