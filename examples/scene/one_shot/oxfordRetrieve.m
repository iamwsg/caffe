%% scenen retrival
clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot


%%prepare files
%fileId=fopen('sameScene.txt');
fileId=fopen('oxford.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

%%
feat=load('oxford.mat');
feat=feat.oxford.posProb;

%% 
load('oxfordQuery.mat');
query=oxfordQuery.queryProb;

%% load the whiter
% w=load('mau_whiter.mat');
% w=w.whiter;

n=size(feat,1);
vname=@(x) inputname(1);

%%
for jj=1:size(query,1)
    disp(jj);
   

    %% prepare test result containner
    
    tRes=cell(n, 5); %14:naive thresholding label; 15:two-step label; 16: first step mark

%% do the test
    for ii=1:n
        
        %disp(ii)
        
        tRes{ii,1}=oxfordQuery.name{jj};
        cat=cats{1}{ii};
        sp=strsplit(cat,'/');
        tRes{ii,2}=sp{7};
%         if strcmp(testScene,qScene)
%             tRes{ii,3}=0;
%         else 
%             tRes{ii,3}=1;
%         end

        r1=feat(jj,:);
        r2=feat(ii,:);
        
        %r1=(w*r1')'/length(r1);
        %r2=(w*r2')'/length(r1);
        
        %dist = r1*r2'/norm(r1)/norm(r2);
        dist=r1*r2';
        
        %dist = smax(r1)*smax(r2)';
        %dist=norm(w*r1'-w*r2');
        %dist = -smax(r1)*log2(smax(r2)')-smax(r2)*log2(smax(r1)');
        
        tRes{ii,4}=dist;
                
    end
    
    fileName=strcat('oxfordRetreveAngle205fast/oxfordRetreve_',oxfordQuery.name{jj},'.mat');
    save(fileName, vname(tRes));
end
