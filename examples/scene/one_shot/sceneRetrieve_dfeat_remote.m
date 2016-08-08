%% scenen retrival
clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

%posFeatPath='/home/shaogang/Datasets/posFeats/';
imageRepos='/home/shaogang/Datasets/scenes/';

%%prepare files
fileId=fopen('sameScene.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

n=length(cats{1});
n_colors=1;
resize=227;
vname=@(x) inputname(1);

%% create net
caffe.reset_all();
model = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
%labelFile='/media/sf_Datasets/placesCNN_upgraded/label205.csv';
resize=227;
caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');
%net.blobs('data').reshape([resize resize 3 2]); % reshape blob 'data'
%net.reshape();

for jj=551:600
    imgPath=cats{1}{jj};
    disp(imgPath)
    cpath=strsplit(imgPath,'/');
    qImagPath=strcat('/home/shaogang/',cpath{4},'/',cpath{5},'/',cpath{6},'/',cpath{7});

%% load query image 
qScene=cpath{6};

qImageCells=strsplit(qImagPath, '/');
nQcell=length(qImageCells);



%% prepair test image pool
testImags=dir(posFeatPath);

%% prepare test result containner
nTest = length(testImags)-2;
tRes=cell(nTest, 13+5); %14:naive thresholding label; 15:two-step label; 16: first step mark


%% do the test
disp('start testing')
for ii=1:nTest
    tic
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
    
    [tRes{ii,15},tRes{ii,16}]=fDenseFeat(net,image_path1,image_path2);
    
   
    time=toc
    tRes{ii,12}=time;
end
    fileName=strcat('imageRetreveDenseFeat/imageRetreve_',cpath{6},'_',cpath{7},'.mat');
    save(fileName, vname(tRes));
end
