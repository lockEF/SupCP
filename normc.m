function [xn]=normc(x);
[n p]=size(x);
xn=x;
for j=1:p
    tempscale=norm(x(:,j),'fro');
    xn(:,j)=x(:,j)/tempscale;
end;