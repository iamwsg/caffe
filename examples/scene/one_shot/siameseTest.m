clear;
close all;
clc;

addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

%%
TEST_DATA_FILE = '/home/shaogang/caffe/data/mnist/t10k-images-idx3-ubyte';
TEST_LABEL_FILE = '/home/shaogang/caffe/data/mnist/t10k-labels-idx1-ubyte';
n = 10000;

% TEST_DATA_FILE = '/home/shaogang/caffe/data/mnist/train-images-idx3-ubyte';
% %TEST_DATA_FILE = '/home/shaogang/caffe/data/mnist/train-images-replace-23-by-01';
% TEST_LABEL_FILE = '/home/shaogang/caffe/data/mnist/train-labels-idx1-ubyte';
% n = 60000;

fid=fopen(TEST_DATA_FILE);
testData=fread(fid,inf,'unsigned char');
fclose(fid);
%testDataReshape=uint8(reshape(testData(17:end),[n,1,28,28]));
testData=uint8(reshape(testData(17:end),[28,28,n,1]));
testData=permute(testData,[3 4 2 1]);

%% visualize
% see=squeeze(testData(3,1,:,:));
% figure,imshow(see);

%%
fid=fopen(TEST_LABEL_FILE);
testLabel=fread(fid,inf,'unsigned char');
fclose(fid);
testLabel=testLabel(9:end);

%% create net
caffe.reset_all();
model = '/home/shaogang/caffe/examples/siamese/mnist_siamese.prototxt';
%weights = '/home/shaogang/caffe/examples/siamese/My_mnist_siamese_0to6_t89_feat_2_sim_iter_1000.caffemodel';
weights = '/home/shaogang/caffe/examples/siamese/mnist_siamese_0to9_feat_2_fine_iter_6000.caffemodel';

caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');
net.blobs('data').reshape([28 28 1 n]); % reshape blob 'data'
net.reshape();
%%
res = net.forward({permute(testData,[4 3 2 1]) *0.00390625});
res=double(res{1,1});

%%
plotCluster(res,testLabel,[0 1 2 3]);
plotCluster(res,testLabel,[7 8 9]);
%plotCluster3D(res,testLabel,[0 1 2 3 4 5 6]);
%plotCluster3D(res,testLabel,[0 1 2 3 4 5 6 7 8 9]);










% %% Whittenning
% ip2=net.blobs('feat').get_data();
% ip2(3,:)=testLabel;
% %lb=find(testLabel==3 | testLabel==2);
% lb=find(testLabel==3);
% r1=ip2(:,lb);
% r=r1(1:2,:);
% mr=mean(r')';
% zr=r-repmat(mr,[1,length(lb)]);
% invr=inv(zr*zr'/length(lb));
% w=sqrtm(invr)*zr;
% var(w')
% var(zr')
% plotCluster(r,r1(3,:),[2 3]);
% plotCluster(w,r1(3,:),[2 3]);

%% find anormaly
% dist=sqrt(sum(w.^2,1));
% out=find(dist > mean(dist)+2*std(dist));
% outIn=lb(out);
% figure;
% for ii=1:length(outIn)
%     see=squeeze(testData(outIn(ii),1,:,:));
%     imshow(see);
%     pause;
% end

%%
% out=find(dist < mean(dist)-1.5*std(dist) );
% outIn=lb(out);
% figure;
% for ii=1:length(outIn)
%     see=squeeze(testData(outIn(ii),1,:,:));
%     imshow(see);
%     pause;
% end

%% show filters
% ly=net.params('conv1',1).get_data();
% figure,subplot(221),imshow(squeeze(ly(:,:,1,1)*256)),subplot(222),...
%     imshow(squeeze(ly(:,:,1,2)*256)),subplot(223),imshow(squeeze(ly(:,:,1,3)*256)),...
%     subplot(224),imshow(squeeze(ly(:,:,1,4)*256));




