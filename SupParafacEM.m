function [B,V,U,se2,Sf,rec]=SupParafacEM(Y,X,R,args)
% EM algorithm to fit the SupCP model:
% X=U X V1 X ... X VK + E 
% U=YB + F 
%
%
% Input
%       Y           n*q full column rank response matrix (necessarily n>=q)
%                  
%        
%       X           n*p1*...*pK design array
%      
%       R          fixed rank of approximation, R<=min(n,p)   
%
%       args       struct parameter with optional additional parameters
%                  args = struct('field1',values1, ...
%                            'field2',values2, ...
%                            'field3',values3) .
%                  The following fields may be specified:
%
%           AnnealIters:  Annealing iterations (default =100)
%
%           fig: binary argument for whether log likelihood should be
%           plotted at each iteration (default = 0)
%
%           ParafacStart:binary argument for whether to initialize with
%           Parafac factorization (default = 0)
%
%           max_niter: maximum number of iterations (default = 1000)
%
%           convg_thres: convergence threshold for difference in log
%           likelihood (default = 10^(-3))
%
%           Sf_diag: whether Sf is diagnal (default =1, diagonal)
%
% Output
%       B           q*r coefficient matrix for U~Y, 
%                   
%       V           list of length K-1. V{k} is a pXr coefficient matrix 
%                    with columns of norm 1
%
%       U           Conditional expectation of U: nXr
%
%       se2         scalar, var(E)
%       Sf          r*r diagonal matrix, cov(F)
%  
%       rec        log likelihood for each iteration
%      
%
% Created by: Eric Lock (elock@umn.edu) and Gen Li

AnnealIters = 100; %%default to 100 annealing iterations
fig = 0 ; % 1=plot likelihood at each iteration; 0=no show
ParafacStart = 0 ; %0 = random start; 1 = parafac start
max_niter=1000; %maximum number of iterations
convg_thres = 10^(-3); % for log likelihood difference  
Sf_diag=1; % 1= diagonal, 0 = not diagonal

%Update args if specified
if nargin > 3 ;  

  if isfield(args,'AnnealIters') ;   
    AnnealIters = getfield(args,'AnnealIters') ; 
  end ;

  if isfield(args,'fig') ;    
    fig = getfield(args,'fig') ; 
  end ;

  if isfield(args,'ParafacStart') ;    
    ParafacStart = getfield(args,'ParafacStart') ; 
  end ;

  if isfield(args,'max_niter') ;    
    max_niter = getfield(args,'max_niter') ; 
  end ;
  
   if isfield(args,'convg_thres') ;    
    convg_thres = getfield(args,'convg_thres') ; 
  end ;
  
  if isfield(args,'Sf_diag') ;   
    Sf_diag = getfield(args,'Sf_diag') ; 
  end ;
end ;

  %  of resetting of input parameters

[n1,q]=size(Y);
m=size(X);
n=m(1);
L=length(m); % number of modes
K=L-1;
p = prod(m(2:L)); % p1*p2*...*pK           % GL: potentially very large 

% Pre-Check
if (n~=n1)
    error('X does not match Y! exit...');
elseif (rank(Y)~=q)
    error('Columns of Y are linearly dependent! exit...');
end;

Index=1:L;
IndexV=1:(L-1);


%%%initialize via parafac
if(ParafacStart)
    Init = parafac(X,R);  % still has randomness in initial value, but in another layer     % GL
    V = {Init{2:L}};
else
  for(l=2:L) V{l-1}=normc(randn(m(l),R)); end
end
Vmat = zeros(p,R); % a very long matrix (p can be very large)
for(r=1:R)
   Temp = TensProd(V,[r]);                                             
   Vmat(:,r)=reshape(Temp,[],1);
end
Xmat = reshape(permute(X,[2:L 1]),[],n);
U=Xmat'*Vmat;
E=X-TensProd({U V{:}});                                                                                  
se2=var(E(:));
B=inv(Y'*Y)*Y'*U;
if Sf_diag
    Sf=diag(diag((1/n)*(U-Y*B)'*(U-Y*B))); % R*R, diagonal
else
    Sf=(1/n)*(U-Y*B)'*(U-Y*B);
end;

   
%%Compute determinant exactly, using Sylvester's determinant theorem
%%https://en.wikipedia.org/wiki/Determinant#Properties_of_the_determinant
% MatForDet = sqrt(Sf)*Vmat'*Vmat*sqrt(Sf)./se2+eye(R); %R X R
MatForDet = (Sf^.5)*Vmat'*Vmat*(Sf^.5)./se2+eye(R); %R X R
%uses woodbury identity and trace properties
logdet_VarMat=2*sum(log(diag(chol(MatForDet))))+p*log(se2); 

ResidMat=Xmat'-Y*B*Vmat'; % n*p

if Sf_diag
    Sfinv=diag(1./diag(Sf));
else
    Sfinv=inv(Sf);    
end;

Trace = (1/se2)*trace(ResidMat*ResidMat')-(1/se2^2)*trace(Vmat'*ResidMat'*ResidMat*Vmat*inv(Sfinv+(1/se2)*(Vmat'*Vmat)));
logl=(-n/2)*(logdet_VarMat)-.5*Trace;
rec=[logl];
if fig
    figure(101);clf;
    plot(1,rec,'o');
    hold on;
    ylabel('log likelihood');
    xlabel('iterations');
end;

niter=1; 
Pdiff = convg_thres+1;
while(niter<=max_niter && (abs(Pdiff)>convg_thres))
    niter=niter+1;
    
    % record last iter
    logl_old=logl;
    se2_old=se2;
    Sf_old=Sf;
    Vmat_old = Vmat;
    V_old=V;
    B_old=B;
             
    %%E step
    
    if Sf_diag
        Sfinv=diag(1./diag(Sf));
    else
        Sfinv=inv(Sf);    
    end;
    weight=inv(Vmat'*Vmat+se2*Sfinv); % r*r
    cond_Mean=(se2*Y*B*Sfinv + Xmat'*Vmat)*weight; % E(U|X), n*r
    U = cond_Mean;
    cond_Var=inv((1/se2)*Vmat'*Vmat+Sfinv); % cov(U(i)|X), r*r
    %%%Add noise to the conditional mean of U.
    %%%Variance of noise is a decreasing percantage of the variance of the
    %%%true conditional mean.
    if(niter<AnnealIters)
        anneal = (AnnealIters-niter)/AnnealIters;
        U = mvnrnd(cond_Mean,anneal*diag(var(U))); 
    end 
    cond_Mean=U;
    cond_quad=n*cond_Var + U'*U; % E(U'U|X), r*r   
    
    %%Estimate V's
    for(l=2:L)
        ResponseMat = reshape(permute(X,[Index(Index~=l) l]),[],m(l));
        PredMat = zeros(prod(m(Index(Index~=l))),R);
        VParams = zeros(prod(m(Index(Index~=l&Index~=1))),R);
        for(r=1:R)
            Temp = TensProd({U V{IndexV~=(l-1)}},[r]);                 
            PredMat(:,r) = reshape(Temp,[],1);
        if(L==3) Temp = V{IndexV~=(l-1)}(:,r);
        else     Temp = TensProd({V{IndexV~=(l-1)}},[r]);               
        end
             VParams(:,r) = reshape(Temp,[],1);
        end
        V{l-1}= ResponseMat'*PredMat/((VParams'*VParams).*cond_quad);
%        V{l-1}=normc(V{l-1});
    end
    
    %estimate B
    B=(Y'*Y)\Y'*U;
    
    %estimate Sf:
    for(r=1:R)
       Temp = TensProd(V,[r]);                                         
        Vmat(:,r)=reshape(Temp,[],1);
    end
    se2=(trace(Xmat'*(Xmat-2*Vmat*cond_Mean')) + ...
        n*trace(Vmat'*Vmat*cond_Var) + trace(cond_Mean*Vmat'*Vmat*cond_Mean'))/(n*p);
    %estimate diagonal entries for covariance:
    if Sf_diag
        Sf=diag(diag((cond_quad + (Y*B)'*(Y*B)- (Y*B)'*cond_Mean- cond_Mean'*(Y*B) )/n));
    else %estimate full covariance
        Sf=(cond_quad + (Y*B)'*(Y*B)- (Y*B)'*cond_Mean- cond_Mean'*(Y*B) )/n;
    end;

    
    

    %%scaling 
    for(l=2:L) V{l-1}=normc(V{l-1}); end
    VmatS=Vmat;
    for(r=1:R)
       Temp = TensProd(V,[r]);                                       
       VmatS(:,r)=reshape(Temp,[],1);
    end
    Bscaling = ones(q,1)*sqrt(sum(Vmat.^2));  
    B  = B.*Bscaling;
    Sfscaling = sqrt(sum(Vmat.^2))'*sqrt(sum(Vmat.^2));
    Sf = Sf.*Sfscaling;
    Vmat=VmatS; 


    %%calc likelihood
    if Sf_diag
        Sfinv=diag(1./diag(Sf));
    else
        Sfinv=inv(Sf);    
    end;                                           

    ResidMat=Xmat'-Y*B*Vmat'; % n*p
     MatForDet = (Sf^.5)*Vmat'*Vmat*(Sf^.5)./se2+eye(R); %R X R
     logdet_VarMat=2*sum(log(diag(chol(MatForDet))))+p*log(se2); 
    Trace = (1/se2)*trace(ResidMat*ResidMat')-...
        (1/se2^2)*trace(Vmat'*ResidMat'*ResidMat*Vmat*inv(Sfinv+(1/se2)*(Vmat'*Vmat)));
    logl=(-n/2)*(logdet_VarMat)-.5*Trace;
    rec=[rec,logl];

    % iteration termination
    Ldiff=logl-logl_old; % should be positive
    if fig
        figure(101);
        if Ldiff<0
            plot(niter,logl,'r*');
        else
            plot(niter,logl,'bo');
        end;
        title(['Current logl: ',num2str(logl,'%10.5e')]);
        drawnow;
    end;

    Pdiff=Ldiff;                                                          
end;



if niter<max_niter
    disp(['EM converges at precision ',num2str(convg_thres),' after ',num2str(niter),' iterations.']);
else
    disp(['EM does not converge at precision ',num2str(convg_thres),' after ',num2str(max_niter),' iterations!!!']);
end;



% re-order parameters

[~,I]=sort(diag(Sf),'descend');
for(k=1:(L-1)) V{k} = V{k}(:,I); end
B=B(:,I);
Sf=Sf(I,I);
U = U(:,I);




