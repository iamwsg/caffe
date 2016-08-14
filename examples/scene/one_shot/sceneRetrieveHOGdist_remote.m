%% scenen retrival HOG
clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

posFeatPath='/home/shaogang/Datasets/posFeatsHOG/';
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
labelFile='/home/shaogang/Downloads/placesCNN_upgraded/categoryIndex_places205.csv';
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% prepair test image pool
testImags=dir(posFeatPath);

%% preload negtive features
disp('preload negative features')
load('/home/shaogang/Datasets/negHOG.mat');
neg=negHOG;

for jj=551:600
    imgPath=cats{1}{jj};
    disp(imgPath)
    cpath=strsplit(imgPath,'/');
    qImagPath=strcat('/home/shaogang/',cpath{4},'/',cpath{5},'/',cpath{6},'/',cpath{7});

%% load query image 
qScene=cpath{6};
%qImagPath=strcat('/media/sf_Datasets/Scenes/images/',qScene,'/1.jpg');
qImag=imread(qImagPath);
%figure, imshow(qImag);

qImageCells=strsplit(qImagPath, '/');
nQcell=length(qImageCells);
qImagFeatFile=[posFeatPath,qImageCells{nQcell-1},'_',qImageCells{nQcell},'.mat']

%feat=load(qImagFeatFile);
%pos_feat1=feat.pos_feat;




%% prepare test result containner
nTest = length(testImags)-2;
tRes=cell(nTest, 13+5); %14:naive thresholding label; 15:two-step label; 16: first step mark
% for ii=1:nTest
%     tRes(ii,1)=tp{1}(ii);
%     tRes(ii,2)=tp{2}(ii);
%     tRes{ii,3}=tp{3}(ii);
% end

resize=227;
blk=16;
%% do the test
disp('start testing')
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
        
    %%coarser classify
    %[p1,p2,r1,r2,feat,cats1,cats2]=coarse_class(net,C,resize,image_path1,image_path2);
    
    im1=imread(image_path1);
    im2=imread(image_path2);
    im1=imresize(im1,[resize,resize]);
    im2=imresize(im2,[resize,resize]);
    r1 = extractHOGFeatures(im1,'BlockSize',[blk blk]);
    r2 = extractHOGFeatures(im2,'BlockSize',[blk blk]);
    v=r1-r2;
    dist = sqrt(v*v')/norm(r1)/norm(r2);
   
    tRes{ii,4}=dist;
    
    
    
end
    fileName=strcat('imageRetreveHOGdist/imageRetreveHOGdist_',cpath{6},'_',cpath{7},'.mat');
    save(fileName, vname(tRes));
end