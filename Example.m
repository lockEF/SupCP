%%Simple simulation for supervised parafac (cp) method

Utrue=20*randn(30,2);
Vtrue=randn(10,2);
Vtrue=normc(Vtrue); 
Wtrue=randn(10,2);
Wtrue=normc(Wtrue);
Ztrue=randn(10,2);
Ztrue=normc(Ztrue);
Xtrue=TensProd({Utrue,Vtrue,Wtrue,Ztrue});
X=  Xtrue+5*randn(30,10,10,10); %%add noise
Y=Utrue(:,1)+randn(30,1); %Y is vector related to one of the signals in U

args = struct('convg_thres',10^(-5),'max_niter',2000,'AnnealIters',1000)
[B,V,U,se2,Sf,rec]=SupParafacEM(Y,X,2,args);

