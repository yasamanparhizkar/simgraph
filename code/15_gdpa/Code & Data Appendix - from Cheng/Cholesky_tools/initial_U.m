function [M] = initial_U(n,mode)

if mode==1
    M=eye(n);
elseif mode==2
    rng(0);
    M=randn(n);
    M=tril(M);
else   
end

end

