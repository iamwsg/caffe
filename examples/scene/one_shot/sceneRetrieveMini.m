%% scenen retrival HOG
clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

posFeatPath='/home/shaogang/caffe/examples/scene/one_shot/posFeatsMini/';
imageRepos='/home/shaogang/Datasets/scenes/';

%%prepare files
fileId=fopen('sameScene.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

n=length(cats{1});
n_colors=1;
resize=227;
vname=@(x) inputname(1);

%% prepare lable file
labelFile='/media/sf_Datasets/placesCNN_upgraded/label205.csv';
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% prepair test image pool
testImags=dir(posFeatPath);

%% preload negtive features
% disp('preload negative features')
% load('/home/shaogang/Datasets/negHOG.mat');
% neg=negHOG;

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

qImageCells=strsplit(qImagPath, '/');
nQcell=length(qImageCells);
qImagFeatFile=[posFeatPath,qImageCells{nQcell-1},'_',qImageCells{nQcell},'.mat']

feat=load(qImagFeatFile);
pos_feat1=feat.pos_feat;


%% prepare test result containner
nTest = length(testImags)-2;
tRes=cell(nTest, 13+5); %14:naive thresholding label; 15:two-step label; 16: first step mark

%% do the test
disp('start testing')
for ii=1:nTest
    %tic
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
    
    tImageCells=strsplit(image_path2, '/');
    nTcell=length(tImageCells);
    tImagFeatFile=[posFeatPath,tImageCells{nTcell-1},'_',tImageCells{nTcell},'.mat'];

    tFeat=load(tImagFeatFile);
    pos_feat2=tFeat.pos_feat;
    pos_feat1=double(pos_feat1);
    pos_feat2=double(pos_feat2);  
    v=pos_feat1-pos_feat2;
    dist = sqrt(v*v')/norm(pos_feat1)/norm(pos_feat2);
   
    tRes{ii,4}=dist;
    
    %time=toc;
    %tRes{ii,12}=time;
end
    fileName=strcat('imageRetreveMini/imageRetreveMini_',cpath{6},'_',cpath{7},'.mat');
    save(fileName, vname(tRes));
end