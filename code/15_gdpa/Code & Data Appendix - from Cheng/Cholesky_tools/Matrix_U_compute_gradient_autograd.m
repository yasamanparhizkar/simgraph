function [ G ] = Matrix_U_compute_gradient_autograd(U,x,N,c)

% e=exp(-sum(c*M.*c,2));
% c=reshape(c', [n 1 N^2]);
% d=-reshape(c.*reshape(c,[1 n N^2]), [n^2 N^2])';
% G_core=e.*y.*d;
% G=reshape(sum(G_core,1), [n n]);

%%======trying autograd begins====
x0=dlarray(U);
[~,G]=dlfeval(@rosenbrock,x0);

% disp(['gradient at point(' num2str(x0(:)') '): ' num2str(G(:)')]);

%%======trying autograd ends======

    function [f,grad]=rosenbrock(variable_x)
        %         c=dlarray(c); % feature difference
        %         x=dlarray(x); % label (can be a vector or a matrix)
        D=dlarray(zeros(N));
        %D_sqrt=dlarray(zeros(N));
        
        W=exp(-sum(c*(variable_x*variable_x').*c,2));
        W(1:N+1:end) = 0;
        W=reshape(W, [N N]);
        sum_W=sum(W);
        %sum_W_sqrt=1./sqrt(sum_W);
        
        D(1:N+1:end)=sum_W;
        %D_sqrt(1:N+1:end)=sum_W_sqrt;
        
        L=D-W;
        %L=D_sqrt*L*D_sqrt;
        L=(L+L')/2;
        %         L0=(L+L')/2;
        %noc=size(x,2);
        %cL=beta_0(1)*L+beta_0(2)*L*L+beta_0(3)*L*L*L;
        f=x'*L*x;
        %f_=x'*cL*x;
        
        %f=sum(f_(1:noc+1:end));
        
        grad=dlgradient(f,variable_x);
    end
end

