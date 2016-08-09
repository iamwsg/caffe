%% scenen retrival
clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

posFeatPath='/home/shaogang/Datasets/posFeats/';
imageRepos='/home/shaogang/Datasets/scenes/';

%%prepare files
fileId=fopen('sameScene.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

n=length(cats{1});
n_colors=1;
resize=227;
vname=@(x) inputname(1);

for jj=559:574
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

load(qImagFeatFile);

%% prepare lable file
labelFile='/home/shaogang/Downloads/placesCNN_upgraded/categoryIndex_places205.csv';
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% prepair test image pool
testImags=dir(posFeatPath);

%% preload negtive features
disp('preload negative features')
negMap = containers.Map;
negfiles=dir('negFeats');
for ii=3:length(negfiles)
    nameCells= strsplit(negfiles(ii).name,'.');
    name=nameCells{1};
    loadNeg=load(['negFeats/' negfiles(ii).name]);
    negMap(name)=loadNeg.feat;
end

%% setup thresholds
posDist=1.6;
posInter=0;

negDist=4;
negInter=0;

thDist=3; %naive threshold
thSim=0.25;

%% prepare test result containner
nTest = length(testImags)-2;
tRes=cell(nTest, 13+5); %14:naive thresholding label; 15:two-step label; 16: first step mark
% for ii=1:nTest
%     tRes(ii,1)=tp{1}(ii);
%     tRes(ii,2)=tp{2}(ii);
%     tRes{ii,3}=tp{3}(ii);
% end

%% create net
caffe.reset_all();
model = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
%labelFile='/media/sf_Datasets/placesCNN_upgraded/label205.csv';
resize=227;
caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');
net.blobs('data').reshape([resize resize 3 2]); % reshape blob 'data'
net.reshape();

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
    
% %     image1=imread(image_path1);
% %     image2=imread(image_path2);
% %     figure,subplot(211),imshow(image1),subplot(212),imshow(image2);
    
    %%coarser classify
    [p1,p2,r1,r2,feat,cats1,cats2]=coarse_class(net,C,resize,image_path1,image_path2);
    v=r1-r2;
    dist = sqrt(v'*v)/norm(r1)/norm(r2);
    catsUnion=union(cats1,cats2);
    catsInter=intersect(cats1,cats2)
    tRes{ii,4}=dist;
    tRes{ii,5}=p1;
    tRes{ii,6}=p2;
    tRes{ii,7}=catsUnion;
    tRes{ii,8}=catsInter;
    tRes{ii,9}=numel(catsInter);
    %%do naive thresholding
    if dist<thDist
        tRes{ii,14}=0;
    else
        tRes{ii,14}=1;
    end
    %% do two step thresholding
    if (dist<posDist) && (numel(catsInter)>=posInter)
        tRes{ii,15}=0;
        tRes{ii,16}=1;
        time=toc;
        tRes{ii,12}=time;
        %continue;
    elseif (dist>negDist) && (numel(catsInter)<=negInter)
        tRes{ii,15}=1;
        tRes{ii,16}=1;
        time=toc;
        tRes{ii,12}=time;
        %continue;
    else
        tRes{ii,16}=0;
    end
    tRes{ii,16}=0;        
    %% finner discriminate
    tImageCells=strsplit(image_path2, '/');
    nTcell=length(tImageCells);
    tImagFeatFile=[posFeatPath,tImageCells{nTcell-1},'_',tImageCells{nTcell},'.mat']

    tFeat=load(tImagFeatFile);
    pos_feat2=tFeat.pos_feat1;

    % get negative
    %tic
    neg=[];
    for jj=1:numel(catsUnion)
        try
            neg1=negMap(catsUnion{jj});
            neg=[neg;neg1];
        catch
            disp('neg feat not found')
            tRes{ii,13}=1;
        end
    end
    %disp('get negative features')
    %toc
    
    %% train two linear SVMs
    %tic
    X1=[pos_feat1; neg];
    X2=[pos_feat2; neg];
    [n_positive,~]=size(pos_feat1);
    Y= -ones(length(neg(:,1))+n_positive,1);Y(1:n_positive)=1;
    SVMModel_linear_1 = fitcsvm(X1,Y);
    SVMModel_linear_2 = fitcsvm(X2,Y);
    disp('train 2 SVMs')

    %% predict
    [label1,score1] = predict(SVMModel_linear_1,pos_feat2);
    [label2,score2] = predict(SVMModel_linear_2,pos_feat1);

    aveScore=(mean(score2(:,2))+mean(score1(:,2)))/2

    similarity= SVMModel_linear_1.Beta'*SVMModel_linear_2.Beta/norm(SVMModel_linear_1.Beta)/norm(SVMModel_linear_2.Beta)
    disp('train 2 SVMs and predict')
    toc
    tRes{ii,10}=aveScore;
    tRes{ii,11}=similarity;
    
    if similarity>thSim
        tRes{ii,15}=0;
    else
        tRes{ii,15}=1;
    end
    time=toc
    tRes{ii,12}=time;
end
    fileName=strcat('imageRetreve/imageRetreve_',cpath{6},'_',cpath{7},'.mat');
    save(fileName, vname(tRes));
end
