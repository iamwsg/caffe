%% sceneRetreve ROC

%clear;
name='imageRetreveDenseFeatAngle/';
files=dir(name);
n=length(files)-2;

for ii=1:n
    disp(strcat(name,files(ii+2).name));
    res=load(strcat(name,files(ii+2).name));
    [pfa(ii,:),pd(ii,:)]=fSceneAnalysisDeepAngle(res.tRes);
end

%%
figure;
for ii=1:n
    disp(ii)
    disp(strcat(name,files(ii+2).name));
    %pause(1);
    plot(pfa(ii,:),pd(ii,:)),hold on;
end
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');grid;

%%
mpfa=mean(pfa);mpd=mean(pd);
figure,plot(mpfa,mpd,'LineWidth',2),grid;title('ROC'),xlabel('P_{fa}'),ylabel('P_d');

%ii=100;figure,plot(pfa(ii,:),pd(ii,:)),grid,disp(strcat(files(ii+2).name));

%mpfa1=mean(pfa);mpd1=mean(pd);
figure,plot(mpfa,mpd,'b',mpfa1,mpd1,'r','LineWidth',2),grid;title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
