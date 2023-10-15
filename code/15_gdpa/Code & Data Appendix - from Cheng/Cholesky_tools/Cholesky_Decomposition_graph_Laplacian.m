function [ L ] = Cholesky_Decomposition_graph_Laplacian( N, c, U)
W=exp(-sum(c*(U*U').*c,2));
W=reshape(W, [N N]);
W(1:N+1:end) = 0;
L = diag(sum(W))-W;
end

