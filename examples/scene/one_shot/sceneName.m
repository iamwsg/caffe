function [sName,imageName] = sceneName(fileName)
    tnames=strsplit(fileName,'.');
    tnames2=strsplit(tnames{1},'_');
    sName=tnames2{1};
    nName=length(tnames2);
    if nName>2
        for jj=2:nName-1
            sName=strcat(sName,'_',tnames2{jj});
        end
    end
    imageName=strcat(tnames2{nName},'.jpg');

end