%% Scene Test

clear;
close all;
clc;

addpath /home/shaogangwang/mywork/caffe/matlab
addpath /home/shaogangwang/mywork/caffe/examples/scene/one_shot

%% prepare train file
disp('prepare test file')
fid=fopen('test1_pairs.txt');
tp=textscan(fid,'%s %s %d');
fclose(fid);

nTest=length(tp{1});


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
nTest = 1000;
tRes=cell(nTest, 13+5); %14:naive thresholding label; 15:two-step label; 16: first step mark
for ii=1:nTest
    tRes(ii,1)=tp{1}(ii);
    tRes(ii,2)=tp{2}(ii);
    tRes{ii,3}=tp{3}(ii);
end
%tRes{:,13}=0;

%% do the test

disp('start training')
for ii=1:nTest
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
    
    %% do naive thresholding
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
        continue;
    elseif (dist>negDist) && (numel(catsInter)<=negInter)
        tRes{ii,15}=1;
        tRes{ii,16}=1;
        time=toc;
        tRes{ii,12}=time;
        continue;
    else
        tRes{ii,16}=0;
    end
            
    %%finner discriminate
    
    %% get positive samples
    %tic
    disp('do second stage discrimination')
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
    
    if similarity>thSim
        tRes{ii,15}=0;
    else
        tRes{ii,15}=1;
    end
    time=toc
    tRes{ii,12}=time;
end


%% seperate positive and negative
load('test1_1000.mat');
nTrain=length(tRes);
%ttRes=tRes(1:nTrain,:);

for ii=1:nTrain
    if isempty(tRes{ii,13})
        tRes{ii,13}=0;
    end
end

ttRes=tRes(find([tRes{:,13}]==0),:); %find valid res
nTrain=length(ttRes);

% invalidUnion=[];
% for ii=length(resInvalid)
%     invalidUnion=union(invalidUnion,resInvalid{ii,7});
% end

trueTruthPos= ttRes(find([ttRes{:,3}]==0),:);
trueTruthNeg= ttRes(find([ttRes{:,3}]==1),:);

naivePos=ttRes(find([ttRes{:,14}]==0),:);
naiveNeg=ttRes(find([ttRes{:,14}]==1),:);

findPos=ttRes(find([ttRes{:,15}]==0),:);
findNeg=ttRes(find([ttRes{:,15}]==1),:);

ap=@(x,label)length(find([x{:,3}]==label))/length(x);

posAP=ap(findPos,0)
negAP=ap(findNeg,1)

posAPnaive=ap(naivePos,0)
negAPnaive=ap(naiveNeg,1)

recall=@(found,ground,label)length(find([found{:,3}]==label))/length(ground);

posRecall=recall(findPos,trueTruthPos,0)
negRecall=recall(findNeg,trueTruthNeg,1)

posRecallNaive=recall(naivePos,trueTruthPos,0)
negRecallNaive=recall(naiveNeg,trueTruthNeg,1)

