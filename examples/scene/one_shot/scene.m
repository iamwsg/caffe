%%scene matching
clear;
close all;
clc;

image_path1='/media/sf_Datasets/Scenes/images/Taj_Mahal/20.jpg';
image_path2='/media/sf_Datasets/Scenes/images/Eiffel_Tower/29.jpg';

addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot
tic
disp('prepare net')
%%create net
caffe.reset_all();
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
toc
%%labels
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% Step1: coasar classification
image1=imread(image_path1);
image2=imread(image_path2);
figure,subplot(211),imshow(image1),subplot(212),imshow(image2);
[p1,p2,r1,r2,feat]=coarse_class(net,C,resize,image_path1,image_path2);
p1',p2'
v=r1-r2;
dist = sqrt(v'*v)/norm(r1)/norm(r2)

toc
disp('coarse classify')
%%need to find the emprical distance bettween same and different scenes and
%%then set a threshold,better to aggragate sematically similar categories, better to
%%use two thresholds, one for match, the other for non-match

%% Step2: For those images with close Euclidean distance in the feature
%%space, find their common category and train two SVM
% negative_file_path='/media/sf_Datasets/Scenes/Taj_Mahal_negative.txt';
%neg_feat=Neg_feat(negative_file_path,net,resize);

%% get positive samples
n_colors=2;
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
toc

net.blobs('data').reshape([resize resize 3 n_positive]); % reshape blob 'data'
net.reshape();
res_pos1 = net.forward({pos1});
pos_feat1 = net.blobs('fc7').get_data()';
res_pos2 = net.forward({pos2});
pos_feat2 = net.blobs('fc7').get_data()';

disp('get positive features')
toc
% for ii=1:375
%     subplot(211),imshow(image1);
%     subplot(212),imshow(ims1(:,:,:,ii));
%     disp(ii);
%     pause(.1);
% end

% get negative
neg_feat1=load('tower_neg_feat.mat');
neg_feat2=load('Taj_Mahal_neg_feat.mat');
neg=[neg_feat1.neg_feat;neg_feat2.neg_feat];
%neg=neg_feat1.neg_feat;
disp('load negative features')
toc

%% train two linear SVMs
X1=[pos_feat1; neg];
X2=[pos_feat2; neg];

Y= -ones(length(neg(:,1))+n_positive,1);Y(1:n_positive)=1;
SVMModel_linear_1 = fitcsvm(X1,Y);
SVMModel_linear_2 = fitcsvm(X2,Y);

disp('train 2 SVMs')
toc

%% visialize positive and negtive samples
[coef,pca_score,latent]=pca(X1);
pp1=pos_feat1*coef(:,1:2);
[coef2,pca_score2,latent2]=pca(X2);
pp2=pos_feat2*coef2(:,1:2);
nn1=neg*coef(:,1:2);
nn2=neg*coef2(:,1:2);
%sup1=SVMModel_linear_1.SupportVectors*coef(:,1:2);
%sup2=SVMModel_linear_2.SupportVectors*coef2(:,1:2);
sup1=pos_feat2*coef(:,1:2);
sup2=pos_feat1*coef2(:,1:2);
figure,subplot(211),plot(pp1(:,1),pp1(:,2),'bo',nn1(:,1),nn1(:,2),'r*',sup1(:,1),sup1(:,2),'blacko'),grid;
subplot(212),plot(pp2(:,1),pp2(:,2),'go',nn2(:,1),nn2(:,2),'r*',sup2(:,1),sup2(:,2),'blacko'),grid;

nn1_1=neg_feat1.neg_feat*coef(:,1:2);nn1_2=neg_feat2.neg_feat*coef(:,1:2);
figure,plot(pp1(:,1),pp1(:,2),'bo',nn1_1(:,1),nn1_1(:,2),'r*',nn1_2(:,1),nn1_2(:,2),'m*',sup1(:,1),sup1(:,2),'blacko'),grid;
nn2_1=neg_feat1.neg_feat*coef2(:,1:2);nn2_2=neg_feat2.neg_feat*coef2(:,1:2);
figure,plot(pp2(:,1),pp2(:,2),'bo',nn2_1(:,1),nn2_1(:,2),'r*',nn2_2(:,1),nn2_2(:,2),'m*'),grid;
%% predict
[label1,score1] = predict(SVMModel_linear_1,pos_feat2);
[label2,score2] = predict(SVMModel_linear_2,pos_feat1);
disp('similarity percentage:');
k1=find(label1==1);
k2=find(label2==1);
length(k1)/n_positive
length(k2)/n_positive
disp('average score:');
ave=(mean(score2(:,2))+mean(score1(:,2)))/2

% ScoreSVMModel1 = fitPosterior(SVMModel_linear_1,X1,Y);
% ScoreSVMModel2 = fitPosterior(SVMModel_linear_2,X2,Y);
% [label3,score3] = predict(ScoreSVMModel1,pos_feat2);
% [label4,score4] = predict(ScoreSVMModel2,pos_feat1);
% disp('average probability:');
% prob=(mean(score3(:,2))+mean(score4(:,2)))/2


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












