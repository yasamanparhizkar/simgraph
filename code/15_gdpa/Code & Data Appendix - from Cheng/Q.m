function [L,A]=Q(n_sample,number_of_neighbor,signal)

A=zeros(n_sample);
for i=1:n_sample
    for j=1:n_sample
        if i<j && j<=i+number_of_neighbor
        A(i,j:min([i+number_of_neighbor n_sample]))=1;
        A(j:min([i+number_of_neighbor n_sample]),i)=A(i,j:min([i+number_of_neighbor n_sample]));
        end
    end
end

D=sum(A);

L=diag(D)-A;

end