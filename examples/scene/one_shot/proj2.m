function [mau2, l_mau]=proj2(mau)

[c_mau,~,l_mau]=pca(mau);
mau2= mau*c_mau(:,1:2);

end