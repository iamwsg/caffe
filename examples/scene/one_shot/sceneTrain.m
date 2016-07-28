%%scene train
clear;
close all;
clc;

addpath /home/shaogangwang/mywork/caffe/matlab
addpath /home/shaogangwang/mywork/caffe/examples/scene/one_shot

%% prepare train file
disp('prepare train file')
fid=fopen('train1_pairs.txt');
tp=textscan(fid,'%s %s %d');
fclose(fid);

nTrain=length(tp{1});

%% prepare the net
disp('prepare net')
caffe.reset_all();
caffe.set_mode_gpu();
model = '/home/shaogangwang/Downloads/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/home/shaogangwang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
net = caffe.Net(model, weights, 'test');
resize=227;
net.blobs('data').reshape([resize resize 3 2]); % reshape blob 'data'
net.reshape();

%% prepare the label file
labelFile='/home/shaogangwang/Downloads/placesCNN_upgraded/categoryIndex_places205.csv';
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% prepare test result containner
tRes=cell(nTrain, 13);
for ii=1:nTrain
    tRes(ii,1)=tp{1}(ii);
    tRes(ii,2)=tp{2}(ii);
    tRes{ii,3}=tp{3}(ii);
end

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

%% do the training
nTrain=3000;
disp('start training')
for ii=1:nTrain
    disp(ii)
    
    image_path1=tRes{ii,1};
    image_path2=tRes{ii,2};
    
%     image1=imread(image_path1);
%     image2=imread(image_path2);
%     figure,subplot(211),imshow(image1),subplot(212),imshow(image2);
    
    tic
    %coarser classify
    [p1,p2,r1,r2,feat,cats1,cats2]=coarse_class(net,C,resize,image_path1,image_path2);
    %p1',p2'
    disp('coarser class time')
    %toc
    
    v=r1-r2;
    dist = sqrt(v'*v)/norm(r1)/norm(r2)
    
    tRes{ii,4}=dist;
    tRes{ii,5}=p1;
    tRes{ii,6}=p2;
    
    catsUnion=union(cats1,cats2);
    catsInter=intersect(cats1,cats2)
    
    tRes{ii,7}=catsUnion;
    tRes{ii,8}=catsInter;
    tRes{ii,9}=numel(catsInter);
    
    %%finner discriminate
    
    %% get positive samples
    %tic
    n_colors=1;
    ims1=Image_aug_color(image_path1,n_colors,resize);
    ims2=Image_aug_color(image_path2,n_colors,resize);
    pos1 = ims1(:, :, [3, 2, 1],:); % convert from RGB to BGR
    pos1 = permute(pos1, [2, 1, 3, 4]); % permute width and height
    pos1 = single(pos1); % convert to single precision
    %imshow(uint8(pos1(:,:,:,1)))
    pos2 = ims2(:, :, [3, 2, 1],:); % convert from RGB to BGR
    pos2 = permute(pos2, [2, 1, 3, 4]); % permute width and height
    pos2 = single(pos2); % convert to single precision
    pos_size=size(pos1);
    n_positive=pos_size(4);
    disp('positive image augmentation')
    %toc

    %tic
    net.blobs('data').reshape([resize resize 3 n_positive]); % reshape blob 'data'
    net.reshape();
    res_pos1 = net.forward({pos1});
    pos_feat1 = net.blobs('fc7').get_data()';
    res_pos2 = net.forward({pos2});
    pos_feat2 = net.blobs('fc7').get_data()';

    %disp('get positive features')
    %toc
    
    net.blobs('data').reshape([resize resize 3 2]); % reshape blob 'data'
    net.reshape();
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
    time=toc
    tRes{ii,12}=time;
end



%% seperate positive and negative
load('train1_3000.mat');
nTrain=length(ttRes);
%ttRes=tRes(1:nTrain,:);

for ii=1:nTrain
    if isempty(ttRes{ii,13})
        ttRes{ii,13}=0;
    end
end

ttRes=ttRes(find([ttRes{:,13}]==0),:); %find valid res
nTrain=length(ttRes);

% invalidUnion=[];
% for ii=length(resInvalid)
%     invalidUnion=union(invalidUnion,resInvalid{ii,7});
% end

resPos= ttRes(find([ttRes{:,3}]==0),:);
resNeg= ttRes(find([ttRes{:,3}]==1),:);

%% statistics of the dist
distPos=cell2mat(resPos(:,4));
distNeg=cell2mat(resNeg(:,4));

distPosMean=mean(distPos)
distPosVar=var(distPos)
distNegMean=mean(distNeg)
distNegVar=var(distNeg)

figure,subplot(211),hist(distPos,100),subplot(212),hist(distNeg,100);

perDistLe=@(x,d)length(find(x<d))/length(x);
perDistGe=@(x,d)length(find(x>d))/length(x);

thDist=2.5;
perDistPos= perDistLe(distPos,thDist)
perDistNeg= perDistGe(distNeg,thDist)
perDistPosFalse=perDistGe(distPos,thDist)
perDistNegFalse=perDistLe(distNeg,thDist)
%% statistics of the intersection classes
interPos=cell2mat(resPos(:,9));
interNeg=cell2mat(resNeg(:,9));

interPosMean=mean(interPos)
interPosVar=var(interPos)
interNegMean=mean(interNeg)
interNegVar=var(interNeg)

figure,subplot(211),hist(interPos,20),subplot(212),hist(interNeg,20);

perInter=@(x,n)length(find(x==n))/length(x);

perInterPos=perInter(interPos,3)
perInterNeg=perInter(interNeg,0)


%% statistics of SVM
svmScorePos=cell2mat(resPos(:,10));
svmScoreNeg=cell2mat(resNeg(:,10));

svmSimPos=cell2mat(resPos(:,11));
svmSimNeg=cell2mat(resNeg(:,11));

figure,subplot(211),hist(svmSimPos,20),subplot(212),hist(svmSimNeg,20);
%figure,subplot(411),hist(svmScorePos,20),subplot(412),hist(svmScoreNeg,20),subplot(413),hist(svmSimPos,20),subplot(414),hist(svmSimNeg,20);

thSim=0.25;
perSvmSimPos=perDistGe(svmSimPos,thSim)
perSvmSimNeg=perDistLe(svmSimNeg,thSim)

%% combination of metrics in step1
distInterPos=[distPos interPos];
distInterNeg=[distNeg interNeg];
%temp=find((distInterNeg(:,2)==0) & (distInterNeg(:,1)>3.5))

th1Neg=@(x,d,n)length(find((x(:,2)<=n) & (x(:,1)>d)))/length(x);
th1Pos=@(x,d,n)length(find((x(:,2)>=n) & (x(:,1)<d)))/length(x);

posDist=1.6;
posInter=0;
perPos1=th1Pos(distInterPos,posDist,posInter)
perFalsePos1=th1Pos(distInterNeg,posDist,posInter)

negDist=4;
negInter=0;
perNeg1=th1Neg(distInterNeg,negDist,negInter)
perFalseNeg1=th1Neg(distInterPos,negDist,negInter)






