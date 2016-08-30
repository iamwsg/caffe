clear;
close all;
abbey=load('/home/shaogang/Datasets/negFeats205/abbey.txt.mat');
airport=load('/home/shaogang/Datasets/negFeats205/airport_terminal.txt.mat');
ocean=load('/home/shaogang/Datasets/negFeats205/ocean.txt.mat');

mabbey=mean(abbey.feat);
mairport=mean(airport.feat);
mocean=mean(ocean.feat);

%%
[c_abbey,s_abbey,l_abbey]=pca(abbey.feat);
abbey2= abbey.feat*c_abbey(:,1:2);

[c_ocean,s_ocean,l_ocean]=pca(ocean.feat);
ocean2= ocean.feat*c_ocean(:,1:2);

[c_airport,s_airport,l_airport]=pca(airport.feat);
airport2= airport.feat*c_airport(:,1:2);

figure,plot(abbey2(:,1),abbey2(:,2),'o',airport2(:,1),airport2(:,2),'ro',ocean2(:,1),ocean2(:,2),'go'),grid,title('origin');
figure,subplot(221),plot(abbey2(:,1),abbey2(:,2),'o'),grid,subplot(222),grid,plot(airport2(:,1),airport2(:,2),'ro'),grid,...
    subplot(223),plot(ocean2(:,1),ocean2(:,2),'go'),grid,title('origin');

%%
dabbey=abbey.feat;
for ii=1:size(abbey.feat,1)
    dabbey(ii,:)=abbey.feat(ii,:)-mabbey;
end

dairport=airport.feat;
for ii=1:size(airport.feat,1)
    dairport(ii,:)=airport.feat(ii,:)-mairport;
end

docean=ocean.feat;
for ii=1:size(ocean.feat,1)
    docean(ii,:)=ocean.feat(ii,:)-mocean;
end

%% demean
[c_abbey,s_abbey,l_abbey]=pca(dabbey);
dabbey2= dabbey*c_abbey(:,1:2);

[c_ocean,s_ocean,l_ocean]=pca(docean);
docean2= docean*c_ocean(:,1:2);

[c_airport,s_airport,l_airport]=pca(dairport);
dairport2= dairport*c_airport(:,1:2);

figure,plot(dabbey2(:,1),dabbey2(:,2),'o',dairport2(:,1),dairport2(:,2),'ro',docean2(:,1),docean2(:,2),'go'),grid,title('demeaned');
figure,subplot(221),plot(dabbey2(:,1),dabbey2(:,2),'o'),grid,subplot(222),grid,plot(dairport2(:,1),dairport2(:,2),'ro'),grid,...
    subplot(223),plot(docean2(:,1),docean2(:,2),'go'),grid,title('demeaned');

%% cov
%conc=[dabbey(1:5000,:); dairport(1:5000,:); docean(1:5000,:)];
conc=[dabbey; dairport; docean];
%conc=docean;
conv=conc'*conc/size(conc,1);
iconv=inv(conv);
siconv=sqrtm(iconv);

%% SVD
[U,S,V]=svd(conc);
siconv=V*1/sqrt(S(1:205,:));


%% whitening
wabbey=abbey.feat*siconv;
wairport=airport.feat*siconv;
wocean=ocean.feat*siconv;

%%
[c_abbey,s_abbey,l_abbey]=pca(wabbey);
wabbey2= wabbey*c_abbey(:,1:2);

[c_ocean,s_ocean,l_ocean]=pca(wocean);
wocean2= wocean*c_ocean(:,1:2);

[c_airport,s_airport,l_airport]=pca(wairport);
wairport2= wairport*c_airport(:,1:2);

figure,plot(wabbey2(:,1),wabbey2(:,2),'o',wairport2(:,1),wairport2(:,2),'ro',wocean2(:,1),wocean2(:,2),'go'),grid,title('whiten');
figure,subplot(221),plot(wabbey2(:,1),wabbey2(:,2),'o'),grid,subplot(222),plot(wairport2(:,1),wairport2(:,2),'ro'),grid,...
    subplot(223),plot(wocean2(:,1),wocean2(:,2),'go'),grid,title('whiten');

%% whiten other sceans
% sky=load('/home/shaogang/Datasets/negFeats4096/sky.txt.mat');
% dsky=sky.feat;
% msky=mean(sky.feat);
% for ii=1:size(sky.feat,1)
%     dsky(ii,:)=sky.feat(ii,:)-msky;
% end
% 
% %%
% wsky=dsky*siconv;
% [c_sky,s_sky,l_sky]=pca(wsky);
% wsky2= wsky*c_sky(:,1:2);
% figure,plot(wsky2(:,1),wsky2(:,2),'mo'),grid;

%%
mau=load('/home/shaogang/Datasets/negFeats205/mausoleum.txt.mat');
mau=mau.feat;

%%
m_mau=mean(mau);
for ii=1:size(mau,1)
    mau(ii,:)=mau(ii,:)-m_mau;
end
[mau2, l_mau]= proj2(mau);
figure,plot(mau2(:,1),mau2(:,2),'o'),grid;

whiter=c_mau*diag(1./sqrt(l_mau));
w_mau= mau*whiter;
[w_mau2,~]=proj2(w_mau);
figure,plot(w_mau2(:,1),w_mau2(:,2),'ro'),grid;


