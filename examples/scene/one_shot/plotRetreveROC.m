%% sceneRetreve ROC

clear;
name='imageRetreveDenseFeat/';
files=dir(name);
n=length(files)-2;

for ii=1:n
    disp(strcat(name,files(ii+2).name));
    res=load(strcat(name,files(ii+2).name));
    [pfa(ii,:),pd(ii,:)]=fSceneAnalysis(res.tRes);
end

%%
figure;
for ii=38:46
    disp(ii)
    disp(strcat(name,files(ii+2).name));
    pause(1.5);
    plot(pfa(ii,:),pd(ii,:)),hold on;
end
grid;

%%
mpfa=mean(pfa);mpd=mean(pd);
figure,plot(mpfa,mpd),grid;

ii=29;figure,plot(pfa(ii,:),pd(ii,:)),grid,disp(strcat(files(ii+2).name));
