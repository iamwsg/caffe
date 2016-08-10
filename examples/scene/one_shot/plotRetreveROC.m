%% sceneRetreve ROC

clear;
files=dir('imageRetreve');
n=length(files)-2;

for ii=5:15
    disp(strcat('imageRetreve/',files(ii+2).name));
    res=load(strcat('imageRetreve/',files(ii+2).name));
    [pfa(ii,:),pd(ii,:)]=fSceneAnalysis(res.tRes);
end

%%
figure;
for ii=38:46
    disp(ii)
    disp(strcat('imageRetreve/',files(ii+2).name));
    pause(1.5);
    plot(pfa(ii,:),pd(ii,:)),hold on;
end
grid;

%%
mpfa=mean(pfa);mpd=mean(pd);
figure,plot(mpfa,mpd),grid;

ii=29;figure,plot(pfa(ii,:),pd(ii,:)),grid,disp(strcat(files(ii+2).name));
