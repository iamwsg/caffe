%% scene analysis
%% seperate positive and negative
clc;
%clear;
close all;
%load('imageRetrieve700_svm_Taj_Mahal_1.mat');

tRes(:,4)=tRes(:,15); %for dense features
%%dist thresholding
dth=0:0.05:4;
%dth=0:0.005:0.6;
dth=linspace(0,0.05); %for dense feat

nTest=length(tRes);
%ttRes=tRes(1:nTest,:);

for ii=1:nTest
    if isempty(tRes{ii,13})
        tRes{ii,13}=0;
    end
end

ttRes=tRes(find([tRes{:,13}]==0),:); %find valid res
nTest=length(ttRes);

% invalidUnion=[];
% for ii=length(resInvalid)
%     invalidUnion=union(invalidUnion,resInvalid{ii,7});
% end


trueTruthPos= ttRes(find([ttRes{:,3}]==0),:);
trueTruthNeg= ttRes(find([ttRes{:,3}]==1),:);

naivePos=ttRes(find([ttRes{:,14}]==0),:);
naiveNeg=ttRes(find([ttRes{:,14}]==1),:);

findPos=ttRes(find([ttRes{:,15}]==0),:);
findNeg=ttRes(find([ttRes{:,15}]==1),:);

ap=@(x,label)length(find([x{:,3}]==label))/length(x);

posAP=ap(findPos,0)
negAP=ap(findNeg,1)

posAPnaive=ap(naivePos,0)
negAPnaive=ap(naiveNeg,1)

recall=@(found,ground,label)length(find([found{:,3}]==label))/length(ground);

posRecall=recall(findPos,trueTruthPos,0)
negRecall=recall(findNeg,trueTruthNeg,1)

posRecallNaive=recall(naivePos,trueTruthPos,0)
negRecallNaive=recall(naiveNeg,trueTruthNeg,1)

%firstStageRes=ttRes(find([ttRes{:,16}]==1),:);
%secondStageRes=ttRes(find([ttRes{:,16}]==0),:);

%aveAcc=@(x)length(find([x{:,3}]==[x{:,15}]))/length(x);
%totalAcc=aveAcc(ttRes)
%firstStageAcc= aveAcc(firstStageRes)
%secondStageAcc= aveAcc(secondStageRes)
%naiveAcc=length(find([ttRes{:,3}]==[ttRes{:,14}]))/length(ttRes)

%falseAlarm=ttRes(find(([ttRes{:,3}]==1) & ([ttRes{:,15}]==0)),:);
%missDetect=ttRes(find(([ttRes{:,3}]==0) & ([ttRes{:,15}]==1)),:);

%falseAlarmRate=length(falseAlarm)/length(find(([ttRes{:,3}]==1)))
%pd=posRecall

%figure,plot(1:nTest, cell2mat(tRes(:,11)),1:nTest, cell2mat(tRes(:,3)),'r',1:nTest, cell2mat(tRes(:,4)),'g'),grid

%% ROC

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
    if ii==52
        kk=52;
    end
    falseAlarm=find(([Res{:,3}]==1) & ([Res{:,15}]==0));
    pfa_dth(ii)=length(falseAlarm)/length(find(([Res{:,3}]==1)));
    pd=find(([Res{:,3}]==0) & ([Res{:,15}]==0));
    mis=find(([Res{:,3}]==0) & ([Res{:,15}]==1));
    pd_dth(ii)= length(pd)/length(find([Res{:,3}]==0));
    ap_dth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,15}]==0));
    recall_dth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,3}]==0));
end
%figure,plot(pfa_dth,pd_dth),grid;

simth=0:1/ndth:1-1/ndth;
%simth=0:0.045/ndth:0.045-0.045/ndth;
pfa_simth=zeros(1,ndth);
pd_simth=zeros(1,ndth);
ap_simth=pd_dth;
recall_simth=pd_dth;
for ii=1:ndth
    Res=tRes;
    for jj=1:nTest
        if Res{jj,11}>simth(ii)
            Res{jj,15}=0;
        else
            Res{jj,15}=1;
        end
    end
    falseAlarm=find(([Res{:,3}]==1) & ([Res{:,15}]==0));
    missDet=find(([Res{:,3}]==0) & ([Res{:,15}]==1));
    pfa_simth(ii)=length(falseAlarm)/length(find(([Res{:,3}]==1)));
    pd=find(([Res{:,3}]==0) & ([Res{:,15}]==0));
    pd_simth(ii)= length(pd)/length(find([Res{:,3}]==0));
    ap_simth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,15}]==0));
    recall_simth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,3}]==0));
end
figure,plot(pfa_dth,pd_dth,'-ob',pfa_simth,pd_simth,'-*r'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
legend('Dist thresholding','SVM thresholding');

% figure,plot(recall_dth,ap_dth,'-ob',recall_simth,ap_simth,'-*r'),grid;
% title('AP-Recall'),xlabel('P_{fa}'),ylabel('P_d');
% legend('Dist thresholding','SVM thresholding');

%pfa2=pfa_dth;
%pd2=pd_dth;
pfa1=pfa_dth;
pd1=pd_dth;


% pfa2=pfa_simth;
% pd2=pd_simth;
% 
figure,plot(pfa1,pd1,'-ob',pfa2, pd2,'-*r'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
legend('BaseLine', 'Deep MatchNet');
%legend('10000 random negatives','Predicted negatives');
