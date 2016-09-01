%% scenen retrival
clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

posFeatPath='/home/shaogang/Datasets/posFeats205/';
negFeatPath='/home/shaogang/Datasets/negFeats205/';
imageRepos='/home/shaogang/Datasets/scenes/';

%%prepare files
fileId=fopen('sameScene.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

%%
feat=load('pos205.mat');
feat=feat.pos205;

%% prepair test image pool
testImags=dir(posFeatPath);

%% load the whiter
w=load('mau_whiter.mat');
w=w.whiter;

n=length(cats{1});
vname=@(x) inputname(1);

%%
for jj=551:600
    imgPath=cats{1}{jj};
    disp(imgPath)
    cpath=strsplit(imgPath,'/');
    qImagPath=strcat('/home/shaogang/',cpath{4},'/',cpath{5},'/',cpath{6},'/',cpath{7});

    %% load query image 
    qScene=cpath{6};
    %qImagPath=strcat('/media/sf_Datasets/Scenes/images/',qScene,'/1.jpg');
    %qImag=imread(qImagPath);
    %figure, imshow(qImag);
    %qImageCells=strsplit(qImagPath, '/');
    %nQcell=length(qImageCells);
    %qImagFeatFile=[posFeatPath,qImageCells{nQcell-1},'_',qImageCells{nQcell},'.mat']
    %load(qImagFeatFile);

    %% prepare test result containner
    nTest = length(testImags)-2;
    tRes=cell(nTest, 5); %14:naive thresholding label; 15:two-step label; 16: first step mark

%% do the test
    for ii=1:nTest
        
        disp(ii)
        image_path1=qImagPath;
        [testScene, testImage]=sceneName(testImags(ii+2).name);
        image_path2=strcat(imageRepos,testScene,'/',testImage);

        tRes{ii,1}=image_path1;
        tRes{ii,2}=image_path2;
        if strcmp(testScene,qScene)
            tRes{ii,3}=0;
        else 
            tRes{ii,3}=1;
        end

        r1=feat(jj,:);
        r2=feat(ii,:);
        
        %r1=(w*r1')'/length(r1);
        %r2=(w*r2')'/length(r1);
        
        %dist = r1*r2'/norm(r1)/norm(r2);
        dist=norm(r1-r2);
        
        %dist = smax(r1)*smax(r2)';
        %dist=norm(w*r1'-w*r2');
        %dist = -smax(r1)*log2(smax(r2)')-smax(r2)*log2(smax(r1)');
        
        tRes{ii,4}=dist;
                
    end
    
    fileName=strcat('imageRetreveAngle205fast/imageRetreve_',cpath{6},'_',cpath{7},'.mat');
    save(fileName, vname(tRes));
end
