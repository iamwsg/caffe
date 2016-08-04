%% gen 10,000 neg CNN features
clear;
%% preload negtive features
disp('preload negative features')
negMap = containers.Map;
negfiles=dir('negFeats');
for ii=3:length(negfiles)
    nameCells= strsplit(negfiles(ii).name,'.');
    name=nameCells{1};
    loadNeg=load(['negFeats/' negfiles(ii).name]);
    negMap(name)=loadNeg.feat;
end

%%
key=keys(negMap)
nkey=length(key);

negFeats10000=zeros(10000,4096);
for ii=1:200
    disp(ii)
    ccat=key(randi(nkey,1));
    if strcmp(ccat{1},'windmill')
        ccat=key(randi(nkey,1));
    end
    cat=negMap(ccat{1});
    [sizecat,~]=size(cat);
    in=randi(sizecat,[50,1]);
    negFeats10000(((ii-1)*50+1:ii*50),:)=cat(in,:);
end


