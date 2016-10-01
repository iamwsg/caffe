function out= rotateDigitMNIST(outSize, d1)
    out=zeros(28,28,outSize);
    inSize=size(d1,3);
    if inSize>outSize
        out=permute(d1(:,:,1:outSize),[2,1,3]);
    else
        out(:,:,1:inSize)=permute(d1,[2,1,3]);
        out(:,:,inSize+1:outSize)=permute(d1(:,:,1:outSize-inSize+1),[2,1,3]);
    end
    
end