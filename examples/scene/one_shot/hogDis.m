%% HOG
clear;

negative_file_path='/media/sf_Datasets/Scenes/Taj_Mahal_negative.txt';
fileID = fopen(negative_file_path);
N = textscan(fileID,'%s %d');
fclose(fileID); 

blk=16;
resize=227;
n_neg = length(N{1});
%% Calculate negative features
% neg_feat=zeros(n_neg,9216);
% for ii=1:n_neg
%     im_path=strcat('/media/sf_Datasets/Scenes/', N{1}(ii));
%     %figure,imshow(imread(im_path{1}));
%     im=imread(im_path{1});
%     im=imresize(im,[resize,resize]);
%     if ndims(im)~=3
%         disp('outlier');
%         disp(im_path);
%         continue;
%     end
%     neg_feat(ii,:) = extractHOGFeatures(im,'BlockSize',[blk blk]);
%     
%     disp(ii);
% end

%% get pos HOG feat
load('Taj_Mahal_neg_feat_HOG.mat');
n_pos=50;
pos_feat=zeros(n_pos,9216);
for ii=1:n_pos
    im_path=strcat('/media/sf_Datasets/Scenes/images/Taj_Mahal/',int2str(ii),'.jpg');
    im=imread(im_path);
    im=imresize(im,[resize,resize]);
    if ndims(im)~=3
        disp('outlier');
        disp(im_path);
        continue;
    end
    pos_feat(ii,:) = extractHOGFeatures(im,'BlockSize',[blk blk]);
    
    disp(ii);
end

%% visiualization
X1=[pos_feat; neg_feat];
[coef,pca_score,latent]=pca(X1);
pp1=pos_feat*coef(:,1:2);
nn1=neg_feat*coef(:,1:2);
figure,plot(pp1(:,1),pp1(:,2),'bo',nn1(:,1),nn1(:,2),'r*'),grid;
title('HOG space'),legend('Positive samples','Hard negative samples');
%% Deep feature
%% prepare the net
disp('prepare net')
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot
caffe.reset_all();
caffe.set_mode_cpu();
model = '/media/sf_Datasets/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/media/sf_Datasets/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
net = caffe.Net(model, weights, 'test');
resize=227;
net.blobs('data').reshape([resize resize 3 1]); % reshape blob 'data'
net.reshape();

%% get deep feature
n_pos=50;
pos_feat=zeros(n_pos,4096);
for ii=1:n_pos
    im_path=strcat('/media/sf_Datasets/Scenes/images/Taj_Mahal/',int2str(ii),'.jpg');
    im=caffe.io.load_image(im_path);
    im = imresize(im, [resize, resize]);
    if ndims(im)~=3
        disp('outlier');
        disp(im_path);
        continue;
    end
    res = net.forward({im});
    pos_feat(ii,:) = net.blobs('fc7').get_data()';
    
    disp(ii);
end

%%
load('Taj_Mahal_neg_feat.mat');
X1=[pos_feat; neg_feat];
[coef,pca_score,latent]=pca(X1);
pp1=pos_feat*coef(:,1:2);
nn1=neg_feat*coef(:,1:2);
figure,plot(pp1(:,1),pp1(:,2),'bo',nn1(:,1),nn1(:,2),'r*'),grid;
title('Deep feature space'),legend('Positive samples','Hard negative samples');



%% image augment and visualization
% n_colors=2;
% ims1=Image_aug_color(im_path,n_colors,resize);
% [~,~,~,n_pos]=size(ims1);
% 
% pos_feat=zeros(n_pos,9216);
% for ii=1:n_pos
%     pos_feat(ii,:) = extractHOGFeatures(ims1(:,:,:,ii),'BlockSize',[blk blk]);
%     disp(ii);
% end
% X1=[pos_feat; neg_feat];
% [coef,pca_score,latent]=pca(X1);
% pp1=pos_feat*coef(:,1:2);
% nn1=neg_feat*coef(:,1:2);
% figure,plot(pp1(:,1),pp1(:,2),'bo',nn1(:,1),nn1(:,2),'r*'),grid;


