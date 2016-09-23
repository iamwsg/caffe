%% scene analysis
function [pfa_dth,pd_dth, scores, labels]=fSceneAnalysisMini(tRes)

nTest=length(tRes);
dth=linspace(0,3e-4); %for dense feat

%% ROC
%%dist thresholding

ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=tRes;
    for jj=1:nTest
        if Res{jj,4}<dth(ii)
            Res{jj,15}=0;
        else
            Res{jj,15}=1;
        end
    end
    falseAlarm=find(([Res{:,3}]==1) & ([Res{:,15}]==0));
    pfa_dth(ii)=length(falseAlarm)/length(find(([Res{:,3}]==1)));
    pd=find(([Res{:,3}]==0) & ([Res{:,15}]==0));
    pd_dth(ii)= length(pd)/length(find([Res{:,3}]==0));
    %ap_dth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,15}]==0));
    %recall_dth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,3}]==0));
end
scores=cell2mat(tRes(:,4));
labels=cell2mat(tRes(:,3));
