%% scene analysis
function [pfa_dth,pd_dth, scores, labels]=fSceneAnalysisDenseFeat(tRes)

nTest=length(tRes);
%ttRes=tRes(1:nTest,:);
tRes(:,4)=tRes(:,15); %for dense features
dth=linspace(0,0.05); %for dense feat

% trueTruthPos= ttRes(find([ttRes{:,3}]==0),:);
% trueTruthNeg= ttRes(find([ttRes{:,3}]==1),:);
% 
% naivePos=ttRes(find([ttRes{:,14}]==0),:);
% naiveNeg=ttRes(find([ttRes{:,14}]==1),:);
% 
% findPos=ttRes(find([ttRes{:,15}]==0),:);
% findNeg=ttRes(find([ttRes{:,15}]==1),:);
% 
% ap=@(x,label)length(find([x{:,3}]==label))/length(x);
% 
% posAP=ap(findPos,0)
% negAP=ap(findNeg,1)
% 
% posAPnaive=ap(naivePos,0)
% negAPnaive=ap(naiveNeg,1)
% 
% recall=@(found,ground,label)length(find([found{:,3}]==label))/length(ground);
% 
% posRecall=recall(findPos,trueTruthPos,0)
% negRecall=recall(findNeg,trueTruthNeg,1)
% 
% posRecallNaive=recall(naivePos,trueTruthPos,0)
% negRecallNaive=recall(naiveNeg,trueTruthNeg,1)

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
%figure,plot(pfa_dth,pd_dth),grid;
scores=cell2mat(tRes(:,4));
labels=cell2mat(tRes(:,3));

