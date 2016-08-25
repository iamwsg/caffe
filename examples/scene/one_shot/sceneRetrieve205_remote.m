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

%% prepare lable file
labelFile='/home/shaogang/Downloads/placesCNN_upgraded/categoryIndex_places205.csv';
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% preload negtive features
disp('preload negative features')
negMap = containers.Map;
negfiles=dir(negFeatPath);
for ii=3:length(negfiles)
    nameCells= strsplit(negfiles(ii).name,'.');
    name=nameCells{1};
    loadNeg=load([negFeatPath negfiles(ii).name]);
    negMap(name)=loadNeg.feat;
end

%% prepair test image pool
testImags=dir(posFeatPath);

%% create net
caffe.reset_all();    %% prepare test result containner
    nTest = length(testImags)-2;
    tRes=cell(nTest, 13+5); %14:naive thresholding label; 15:two-step label; 16: first step mark

%% do the test
    for ii=1:nTest
model = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
%labelFile='/media/sf_Datasets/placesCNN_upgraded/label205.csv';
resize=227;
caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');
net.blobs('data').reshape([resize resize 3 2]); % reshape blob 'data'
net.reshape();

n=length(cats{1});
vname=@(x) inputname(1);

%%
for jj=551:551
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

    %% prepare test result containner
    nTest = length(testImags)-2;
    tRes=cell(nTest, 13+5); %14:naive thresholding label; 15:two-step label; 16: first step mark

%% do the test
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
        similarity= SVMModel_linear_1.Beta'*SVMModel_linear_2.Beta/norm(SVMModel_linear_1.Beta)/norm(SVMModel_linear_2.Beta)
        toc
        tRes{ii,11}=similarity;
        time=toc
        tRes{ii,12}=time;
    end
    
    fileName=strcat('imageRetreve205/imageRetreve_',cpath{6},'_',cpath{7},'.mat');
    save(fileName, vname(tRes));
end
