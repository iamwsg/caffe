function out= rotateMergeDigitMNIST(outSize, d1,d2,d2In)
    out=zeros(28,28,outSize);
    inSize=size(d1,3);
    d=squeeze(permute(d2(:,:,d2In),[2,1,3]));
    dd=repmat(d,1,1,outSize);
    
    if inSize>outSize
        out=permute(d1(:,:,1:outSize),[2,1,3])+ dd;
    else
        out(:,:,1:inSize)=permute(d1,[2,1,3])+repmat(d,1,1,inSize);
        out(:,:,inSize+1:outSize)=permute(d1(:,:,1:outSize-inSize),[2,1,3])+repmat(d,1,1,outSize-inSize);
    end
    
end