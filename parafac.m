function [U,SqError]=parafac(Y,R)
%Performs parafac factorization via ALS
%
% Input
%       Y           Array of dimension m1 X m2 X ... X mK 
%       R           Desired rank of the factorization  
%                   Default is all columns (range = [1:R]).
%
% Output
%       U           List of basis vectors for parafac factorization, 
%                   U{k}: mk X R  for k=1,...,K. Columns of U{K} have norm
%                   1 for k>1.  TensProd(U) approximates Y
%      
%       SqError     Vector of squared error for the approximation at each
%                   iteration (should be non-increasing).  
%
% Created by: Eric F. Lock (elock@umn.edu) and Gen Li
%


m=size(Y);
L=length(m);
U = {};
for(l=2:L) U{l}=normc(randn(m(l),R)); end
Index = 1:L;

SqError = [];
i=0;
thresh = 10^(-1);
SqErrorDiff = thresh+1;
Yest = zeros(m);
iters = 1000;
while(i<iters&&SqErrorDiff>thresh)
    
    i=i+1;
    Yest_old = Yest;
    for(l=1:L)
        ResponseMat = reshape(permute(Y,[Index(Index~=l) l]),[],m(l));
        PredMat = zeros(prod(m(Index(Index~=l))),R);
        for(r=1:R)
            Temp = TensProd({U{Index~=l}},[r]);
            PredMat(:,r) = reshape(Temp,[],1);
        end
        U{l}= ResponseMat'*PredMat*inv(PredMat'*PredMat);


        if(l>1) U{l}=normc(U{l});end %%standardize for mode l>1

    end  

Yest = TensProd(U); 

Error = Y-Yest;
SqError(i) = sum(Error(:).^2);
ErrorDiff = Yest-Yest_old;
SqErrorDiff = sum(ErrorDiff(:).^2);
end
