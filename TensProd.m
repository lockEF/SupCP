function [Ans]=TensProd(L,range)
% Compute tensor product over multiple dimensions
% using Khatri-Rao product
%
% Input
%       L           List of length K, with matrix entries L{k}: m_k X R; 
%       range       Column indices over which to apply tensor product.  
%                   Default is all columns (range = [1:R]).
%
% Output
%       Ans         Array of size m_1 X m_2 ... X m_K, where the [i1,...iK]  
%                   entry is the sum of product L{k}(i1,r)*...*L{K}(iK,r) 
%                   over all r in range.  
%
% Created by: Gen Li

K = numel(L);
m = ones(1,K);
for i=1:K 
    [m(i),R] = size(L{i}); 
end;
if nargin==2 % need to customize L
    % check
    if max(range)>R
        error('Range exceeds rank!');
    end;
    % customize
    newL=L;
    for i=1:K
        newL{i}=L{i}(:,range);
    end;
    L=newL;
end

tempL=L(K:-1:2);
matX=L{1}*kr(tempL)'; % X_(1)
Ans=reshape(matX,m);
end



