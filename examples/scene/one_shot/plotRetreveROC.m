%% sceneRetreve ROC

%clear;
name='imageRetreveAngle205fast/';
files=dir(name);
n=length(files)-2;

for ii=1:n
    disp(strcat(name,files(ii+2).name));
    res=load(strcat(name,files(ii+2).name));
    [pfa(ii,:),pd(ii,:)]=fSceneAnalysisDeepDist(res.tRes);
end

%%
figure;
for ii=1:n
    disp(ii)
    disp(strcat(name,files(ii+2).name));
    sp=strsplit(files(ii+2).name,'.');
    sp=strsplit(sp{1},'_');
    imgPath=strcat('/home/shaogang/Datasets/scenes/','Taj_Mahal/',sp{4},'.jpg');
    %subplot(211),imshow(imread(imgPath)),subplot(212),plot(pfa(ii,:),pd(ii,:)),grid;
    %pause;
    pause(.5);
    plot(pfa(ii,:),pd(ii,:)),hold on;
end
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');grid;

%%
mpfa=mean(pfa);mpd=mean(pd);
figure,plot(mpfa,mpd,'LineWidth',2),grid;title('205 Dist'),xlabel('P_{fa}'),ylabel('P_d');

PdPfa.Pd=mpd;
PdPfa.Pfa=mpfa;

%ii=26;figure,plot(pfa(ii,:),pd(ii,:)),grid,disp(strcat(files(ii+2).name));

%mpfa1=mean(pfa);mpd1=mean(pd);
%figure,plot(mpfa,mpd,'b',mpfa1,mpd1,'r','LineWidth',2),grid;title('ROC'),xlabel('P_{fa}'),ylabel('P_d');

%%
prob=load('PdPfa/prob205.mat');
prob=prob.PdPfa;
cos=load('PdPfa/cos205.mat');
cos=cos.PdPfa;
denseProb=load('PdPfa/denseProb205.mat');
denseProb=denseProb.PdPfa;
eclidean=load('PdPfa/Eclidean205.mat');
eclidean=eclidean.PdPfa;
cross=load('PdPfa/cross205.mat');
cross=cross.PdPfa;

figure,plot(eclidean.Pfa,eclidean.Pd,'b',cos.Pfa,cos.Pd,'g',cross.Pfa,cross.Pd,'m',...
    prob.Pfa,prob.Pd,'black',denseProb.Pfa,denseProb.Pd,'r','LineWidth',2),grid;title('ROC'),xlabel('P_{fa}'),ylabel('P_d');








