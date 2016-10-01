function out= mergeDigitMNIST(Size, d1,d2)
    out=zeros(28,28,Size);
    for ii=1:Size
        u=randi(size(d1,3),1);
        v=randi(size(d2,3),1);
        out(:,:,ii)=d1(:,:,u)+d2(:,:,v);
    end
end