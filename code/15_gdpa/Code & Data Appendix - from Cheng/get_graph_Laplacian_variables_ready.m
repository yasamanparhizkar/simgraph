function [c,y] = get_graph_Laplacian_variables_ready(feature,x,N,n)
y=(x-x.').^2;
y=reshape(y,[N^2 1]);
a=reshape(feature,[N 1 n]);
c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
end
