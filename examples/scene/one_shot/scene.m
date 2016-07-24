%%scene matching
clear;
close all;
clc;

addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot
%%create net
model = '/media/sf_Datasets/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/media/sf_Datasets/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
labelFile='/media/sf_Datasets/placesCNN_upgraded/label205.csv';
resize=227;

% model='/media/sf_Datasets/placesCNN_upgraded/deploy_vgg16_places365.prototxt';
% weights='/media/sf_Datasets/placesCNN_upgraded/vgg16_places365.caffemodel';
% labelFile='/media/sf_Datasets/placesCNN_upgraded/categories_places365.txt';
% resize=224;

caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');

%%labels
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% Step1: coasar classification
image_path1='/media/sf_Datasets/Scenes/images/Sphinx/34.jpg';
image_path2='/media/sf_Datasets/Scenes/images/Sphinx/26.jpg';

image1=imread(image_path1);
image2=imread(image_path2);
figure,subplot(211),imshow(image1),subplot(212),imshow(image2);
[p1,p2,r1,r2,feat]=coarse_class(net,C,resize,image_path1,image_path2);
p1',p2'
v=r1-r2;
dist = sqrt(v'*v)/norm(r1)/norm(r2)

%%need to find the emprical distance bettween same and different scenes and
%%then set a threshold,better to aggragate sematically similar categories, better to
%%use two thresholds, one for match, the other for non-match

%% Step2: For those images with close Euclidean distance in the feature
%%space, find their common category and train two SVM
% negative_file_path='/media/sf_Datasets/Scenes/Taj_Mahal_negative.txt';
%neg_feat=Neg_feat(negative_file_path,net,resize);

%% get positive samples
ims1=Image_aug_color(image_path1,2,resize);
ims2=Image_aug_color(image_path2,2,resize);
pos1 = ims1(:, :, [3, 2, 1],:); % convert from RGB to BGR
pos1 = permute(pos1, [2, 1, 3, 4]); % permute width and height
pos1 = single(pos1); % convert to single precision
%imshow(uint8(pos1(:,:,:,1)))
pos2 = ims2(:, :, [3, 2, 1],:); % convert from RGB to BGR
pos2 = permute(pos2, [2, 1, 3, 4]); % permute width and height
pos2 = single(pos2); % convert to single precision
pos_size=size(pos1);
N_channel=pos_size(4);
net.blobs('data').reshape([resize resize 3 N_channel]); % reshape blob 'data'
net.reshape();
res_pos1 = net.forward({pos1});
pos_feat1 = net.blobs('fc7').get_data()';
res_pos2 = net.forward({pos2});
pos_feat2 = net.blobs('fc7').get_data()';

% for ii=1:375
%     subplot(211),imshow(image1);
%     subplot(212),imshow(ims1(:,:,:,ii));
%     disp(ii);
%     pause(.1);
% end

%% train two SVMs
% get negative
neg_feat=load('Taj_Mahal_neg_feat.mat');
neg=neg_feat.neg_feat;


%% train two linear SVMs
X1=[pos_feat1; neg];
X2=[pos_feat2; neg];
Y= -ones(length(neg(:,1))+N_channel,1);Y(1:N_channel)=1;
SVMModel_linear_1 = fitcsvm(X1,Y);
SVMModel_linear_2 = fitcsvm(X2,Y);

[label1,score1] = predict(SVMModel_linear_1,pos_feat2);
[label2,score2] = predict(SVMModel_linear_2,pos_feat1);
disp('similarity percentage');
k1=find(label1==1);
k2=find(label2==1);
length(k1)/N_channel
length(k2)/N_channel

%% train two kernel SVMs

% SVMModel_RBF_1 = fitcsvm(X1,Y,'Standardize',true,'KernelFunction','RBF',...
%     'KernelScale','auto');
% CVSVMModel = crossval(SVMModel_RBF_1);
% classLoss = kfoldLoss(CVSVMModel)
% 
% SVMModel_RBF_2 = fitcsvm(X2,Y,'Standardize',true,'KernelFunction','RBF',...
%     'KernelScale','auto');
% CVSVMModel = crossval(SVMModel_RBF_1);
% classLoss = kfoldLoss(CVSVMModel)
% 
% [label3,score3] = predict(SVMModel_RBF_1,pos_feat2)
% [label4,score4] = predict(SVMModel_RBF_2,pos_feat1)












