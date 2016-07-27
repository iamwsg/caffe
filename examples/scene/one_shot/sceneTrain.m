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
disp('start training')
for ii=1:3000
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
