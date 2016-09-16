%% sceneRetreve ROC

%clear;
name='oxfordRetreveAngle205fast/';
files=dir(name);
n=length(files)-2;

for ii=1:n
    disp(ii)
    disp(strcat(name,files(ii+2).name));
    res=load(strcat(name,files(ii+2).name));
    %eleminate junk
    ttRes=res.tRes(find([res.tRes{:,3}]~=2),:); %find valid res
    
    [pfa(ii,:),pd(ii,:), recall(ii,:), precision(ii,:)]=fSceneAnalysisDeepDist(ttRes);
end

%%
figure;
for ii=1:n
    disp(ii)
    disp(strcat(name,files(ii+2).name));
    sp=strsplit(files(ii+2).name,'.');
    sp=strsplit(sp{1},'_');
    %imgPath=strcat('/home/shaogang/Datasets/scenes/','Taj_Mahal/',sp{4},'.jpg');
    %subplot(211),imshow(imread(imgPath)),subplot(212),plot(pfa(ii,:),pd(ii,:)),grid;
    %subplot(211),imshow(imread(imgPath)),subplot(212),plot(recall(ii,:),precision(ii,:)),grid;
    %pause;
    
    pause(.1);
    %plot(pfa(ii,:),pd(ii,:)),hold on;
    plot(recall(ii,:),precision(ii,:)),hold on;
end
%title('ROC'),xlabel('P_{fa}'),ylabel('P_d');grid;
title('Recall-Precision'),xlabel('Recall'),ylabel('Precision');grid;

%%
% mpfa=mean(pfa);mpd=mean(pd);
% figure,plot(mpfa,mpd,'LineWidth',2),grid;title('205 Dist'),xlabel('P_{fa}'),ylabel('P_d');
mrecall=mean(recall);mprecise=mean(precision);
figure,plot(mrecall,mprecise,'LineWidth',2),grid;title('Ave Recall-Precision'),xlabel('Recall'),ylabel('Precision');

mpfa=mean(pfa);mpd=mean(pd);
figure,plot(mpfa,mpd,'LineWidth',2),grid;title('Ave ROC'),xlabel('P_{fa}'),ylabel('P_d');

%%mAP
mAP=0;
for ii=2:length(mrecall)
    mAP=mAP+ (mrecall(ii+1)-mrecall(ii))*mprecise(ii);
end


PdPfa.Pd=mpd;
PdPfa.Pfa=mpfa;

%ii=26;figure,plot(pfa(ii,:),pd(ii,:)),grid,disp(strcat(files(ii+2).name));

%mpfa1=mean(pfa);mpd1=mean(pd);
%figure,plot(mpfa,mpd,'b',mpfa1,mpd1,'r','LineWidth',2),grid;title('ROC'),xlabel('P_{fa}'),ylabel('P_d');

%%
% prob=load('PdPfa/prob205.mat');
% prob=prob.PdPfa;
% cos=load('PdPfa/cos205.mat');
% cos=cos.PdPfa;
% denseProb=load('PdPfa/denseProb205.mat');
% denseProb=denseProb.PdPfa;
% eclidean=load('PdPfa/Eclidean205.mat');
% eclidean=eclidean.PdPfa;
% cross=load('PdPfa/cross205.mat');
% cross=cross.PdPfa;
% 
% figure,plot(eclidean.Pfa,eclidean.Pd,'b',cos.Pfa,cos.Pd,'g',cross.Pfa,cross.Pd,'m',...
%     prob.Pfa,prob.Pd,'black',denseProb.Pfa,denseProb.Pd,'r','LineWidth',2),grid;title('ROC'),xlabel('P_{fa}'),ylabel('P_d');



%%
% imgRoot='/home/shaogang/Downloads/oxford/oxbuild_images/'
% name='oxfordRetreveAngle205fast/';
% ii=26;
% files=dir(name);
% n=length(files)-2;
% disp(strcat(name,files(ii+2).name));
% res=load(strcat(name,files(ii+2).name));
% 
% [S, in]=sort([res.tRes{:,4}],'descend');
% tttRes=res.tRes(in,2);
% figure;
% for ii=1:30
%     imshow(imread(strcat(imgRoot,tttRes{ii},'.jpg')));
%     pause;
% end


