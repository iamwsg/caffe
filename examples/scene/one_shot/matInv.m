%% preload negtive features
clc;
clear;
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
keys = keys(negMap);
mat=[];
for ii=1:length(keys)
    mat = [mat; negMap(keys{ii})];
end

%%
m=mean(mat);
%%
[len, ~]=size(mat);
for ii=1:len
    mat(ii,:)=mat(ii,:)-m;
end

%%
[coef,pca_score,latent]=pca(mat);
%%
con=mat'*mat;
icon=inv(con);
%%
negPCA.mean=m;
negPCA.coef=coef;
negPCA.latent=latent;
negPCA.con=con;
negPCA.icon=icon;
