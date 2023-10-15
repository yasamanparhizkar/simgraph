 function [y,obj]=gdpa_core_tsp_test_uc( ...
    n,...
    scaled_M,...
    ddL,...
    scaled_factors,...
    rho,...
    db,...
    options,...
    cL,...
    b_ind,...
    epsilon,...
    dz_ind_plus,...
    dz_ind_minus)

lbi=length(b_ind); % number of known labels

% scaled_M_n=scaled_M(1:n,1:n);
scaled_M(1:n+2+1:end)=0;

% %% check objective lower bound (not exactly the acutal lower bound)
% save_entry_one_by_one=zeros(lbi+1,1);
% save_entry_one_by_one(end)=sum(sum(abs(scaled_M_n),2)-diag(cL));
% for i=1:lbi
%     if sign(db(i))==1 % b_i>0, i.e., z_i<0
%         save_entry_one_by_one(i)=abs(scaled_factors(i,n+1))+abs(scaled_factors(n+1,i))-2;
%     else                  % b_i<0, i.e., z_i>0
%         save_entry_one_by_one(i)=abs(scaled_factors(i,n+2))+abs(scaled_factors(n+2,i))-2;
%     end
% end

%% define [c] [P] [d] in Eq. \eqref{eq:LP_std1} in paper
c=[ones(n+1,1)]; % N+1+M
% d=zeros(n+2,1); % N+2+M
d=-sum(abs(scaled_M),2);
% d(1:n)=diag(cL)-sum(abs(scaled_M),2);
% lb=zeros(n+1+lbi,1)-Inf;
% ub=zeros(n+1+lbi,1)+Inf;
%%========================
d=d-rho;
% d(1:n+2)=d(1:n+2)-rho; % ensure PD/PSD
d(n+1)=d(n+1)-epsilon;
d(n+2)=d(n+2)+epsilon;
%%========================

%% ======define P starts=======
P=zeros(n+2,n+1);
% the first n rows
P(1:n,1:n)=-eye(n); % corresponds to the variables y's (the first n entries)

% for lbi_i=1:lbi
%     if sign(db(lbi_i))==1 % b_i>0, i.e., z_i<0
%         ub(n+1+lbi_i)=-0;
%         scalars=-abs(scaled_factors(lbi_i,n+1));   
%         % the n+1 th row
%         P(n+1,n+1+lbi_i)=-abs(scaled_factors(n+1,lbi_i))+1-0.5;  % X'_{1xM}
%         P(n+2,n+1+lbi_i)=-0.5;
%     else                  % b_i<0, i.e., z_i>0
%         lb(n+1+lbi_i)=0;
%         scalars=abs(scaled_factors(lbi_i,n+2));
%         % the n+2 th row
%         P(n+2,n+1+lbi_i)=abs(scaled_factors(n+2,lbi_i))+1-0.5; % Y'_{1xM}
%         P(n+1,n+1+lbi_i)=-0.5;
%     end
%     P(lbi_i,n+1+lbi_i)=scalars; % E_{NxM}
% end
% the n+1 th row
P(n+1,n+1)=-0.5; % corresponds to the variables y's (the last one entry)
P(n+2,n+1)=-0.5; % corresponds to the variables y's (the last one entry)
% % the next lbi rows
% P(n+1+1:n+1+lbi,n+1+1:n+1+lbi)=eye(lbi);
% P(n+1+1:n+1+lbi,n+1+lbi+1:n+1+2*lbi)=-eye(lbi);
% % the last lbi rows
% P(n+1+lbi+1:n+1+2*lbi,n+1+1:n+1+lbi)=-eye(lbi);
% P(n+1+lbi+1:n+1+2*lbi,n+1+lbi+1:n+1+2*lbi)=-eye(lbi);
%% ======define P ends=======

[s_k,obj] = linprog(c,...
    P,d,...
    [],[],...
    [],[],options);

% s_k=s_k(1:n+1+lbi);
y=s_k(1:n+1);
% z=s_k(n+1+1:n+1+lbi);
end