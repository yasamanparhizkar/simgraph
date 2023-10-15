function [ L ] = graph_Laplacian( N, c, M)
W=exp(-sum(c*M.*c,2));
W=reshape(W, [N N]);
W(1:N+1:end) = 0;
D=diag(sum(W));
L=D-W;
L=D^(-0.5)*L*D^(-0.5);
L=(L+L')/2;
% L=L/max(eig(L));
end

