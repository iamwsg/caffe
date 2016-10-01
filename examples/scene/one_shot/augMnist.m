clear;
close all;
clc;

addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

%%
% TEST_DATA_FILE = '/home/shaogang/caffe/data/mnist/t10k-images-idx3-ubyte';
% TEST_LABEL_FILE = '/home/shaogang/caffe/data/mnist/t10k-labels-idx1-ubyte';
% n = 10000;

TRAIN_DATA_FILE = '/home/shaogang/caffe/data/mnist/train-images-idx3-ubyte';
TRAIN_LABEL_FILE = '/home/shaogang/caffe/data/mnist/train-labels-idx1-ubyte';

%% open image
fp = fopen(TRAIN_DATA_FILE, 'rb');
assert(fp ~= -1, ['Could not open ', TRAIN_DATA_FILE, '']);
magic1 = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic1 == 2051, ['Bad magic number in ', TRAIN_DATA_FILE, '']);
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);
fclose(fp);

%% open label
fp = fopen(TRAIN_LABEL_FILE, 'rb');
assert(fp ~= -1, ['Could not open ', TRAIN_LABEL_FILE, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', TRAIN_LABEL_FILE, '']);
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp);

i0=images(:,:,find(labels==0));
i1=images(:,:,find(labels==1));
i2=images(:,:,find(labels==2));
i3=images(:,:,find(labels==3));
i4=images(:,:,find(labels==4));
i5=images(:,:,find(labels==5));
i6=images(:,:,find(labels==6));
i7=images(:,:,find(labels==7));
i8=images(:,:,find(labels==8));
i9=images(:,:,find(labels==9));

%%
%%%%%%%%%%%%%%% Replace 4 5 6 by combination of 0 1 2 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate artificial 4 5 6 
% i4_01 = mergeDigitMNIST(size(i4,3),i0,i1);
% i5_12 = mergeDigitMNIST(size(i5,3),i1,i2);
% i6_03 = mergeDigitMNIST(size(i6,3),i0,i3);
% 
% images(:,:,find(labels==4))=i4_01;
% images(:,:,find(labels==5))=i5_12;
% images(:,:,find(labels==6))=i6_03;
% 
% images = permute(images,[2 1 3]);
% figure,imshow(images(:,:,1));
% 
% newStream=reshape(images,numel(images),1);
% 
% %% write new data file
% newfile= '/home/shaogang/caffe/data/mnist/train-images-replace-456';
% fid=fopen(newfile,'wb');
% fwrite(fid,magic1,'int32', 'ieee-be');
% fwrite(fid,numImages,'int32', 'ieee-be');
% fwrite(fid,numRows,'int32', 'ieee-be');
% fwrite(fid,numCols,'int32', 'ieee-be');
% fwrite(fid,newStream,'unsigned char');
% fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%%%%%%%%%%%%%%% Replace 4 5 6 by rotation of 0 1 2 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate artificial 4 5 6 
% i4_r1 = rotateDigitMNIST(size(i4,3),i1);
% i5_r2 = rotateDigitMNIST(size(i5,3),i2);
% i6_r3 = rotateDigitMNIST(size(i6,3),i3);
% 
% images(:,:,find(labels==4))=i4_r1;
% images(:,:,find(labels==5))=i5_r2;
% images(:,:,find(labels==6))=i6_r3;
% 
% images = permute(images,[2 1 3]);
% newStream=reshape(images,numel(images),1);
% %%figure,imshow(i6_r3(:,:,15));
 
%%
%%%%%%%%%%%%%%% Replace 4 5 6 by rotation and merge
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate artificial 4 5 6 
% i4_r0_m1 = rotateMergeDigitMNIST(size(i4,3),i0,i1,1);
% i5_r2_m1 = rotateMergeDigitMNIST(size(i5,3),i2,i1,2);
% i6_r3 = rotateDigitMNIST(size(i6,3),i3);
% 
% images(:,:,find(labels==4))=i4_r0_m1;
% images(:,:,find(labels==5))=i5_r2_m1;
% images(:,:,find(labels==6))=i6_r3;
% 
% images = permute(images,[2 1 3]);
% newStream=reshape(images,numel(images),1);
%%figure,imshow(i6_r3(:,:,15));


%%
%%%%%%%%%%%%%%% Replace 4 5 6 7 8 9 by rotation and merge of 0123
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% generate artificial 4 5 6 
% i4_r0_m1 = rotateMergeDigitMNIST(size(i4,3),i0,i1,1);
% i5_r2_m1 = rotateMergeDigitMNIST(size(i5,3),i2,i1,2);
% i6_r3 = rotateDigitMNIST(size(i6,3),i3);
% i7_r1_m3 = rotateMergeDigitMNIST(size(i7,3),i1,i3,2);
% i8_r2 = rotateDigitMNIST(size(i8,3),i2);
% i9_m12 = mergeDigitMNIST(size(i9,3),i1,i2);
% 
% images(:,:,find(labels==4))=i4_r0_m1;
% images(:,:,find(labels==5))=i5_r2_m1;
% images(:,:,find(labels==6))=i6_r3;
% images(:,:,find(labels==7))=i7_r1_m3;
% images(:,:,find(labels==8))=i8_r2;
% images(:,:,find(labels==9))=i9_m12;
% 
% images = permute(images,[2 1 3]);
% newStream=reshape(images,numel(images),1);
%figure,imshow(i6_r3(:,:,15));

%%
% %%%%%%%%%%%%%%% Replace 4 5 6 7 8 9 by rotation and merge of 0123
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% generate artificial 4 5 6 
% i4_r0_m1 = rotateMergeDigitMNIST(size(i4,3),i0,i1,1);
% i5_r2_m1 = rotateMergeDigitMNIST(size(i5,3),i2,i1,2);
% i6_r3 = rotateDigitMNIST(size(i6,3),i3);
% i7_r1_m3 = rotateMergeDigitMNIST(size(i7,3),i1,i3,2);
% i8_r2 = rotateDigitMNIST(size(i8,3),i2);
% i9_m12 = mergeDigitMNIST(size(i9,3),i1,i2);
% 
% images(:,:,find(labels==4))=i4_r0_m1;
% images(:,:,find(labels==5))=i5_r2_m1;
% images(:,:,find(labels==6))=i6_r3;
% images(:,:,find(labels==7))=i7_r1_m3;
% images(:,:,find(labels==8))=i8_r2;
% images(:,:,find(labels==9))=i9_m12;
% 
% images = permute(images,[2 1 3]);
% newStream=reshape(images,numel(images),1);
% %figure,imshow(i6_r3(:,:,15));


%%
%%%%%%%%%%%%%%% Replace 4 5 6 7 8 9 by random images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate artificial 4 5 6 
% i4_r0_m1 = uint8(randi(256,28,28,size(i4,3))-1);
% i5_r2_m1 = uint8(randi(256,28,28,size(i5,3))-1);
% i6_r3 = uint8(randi(256,28,28,size(i6,3))-1);
% i7_r1_m3 = uint8(randi(256,28,28,size(i7,3))-1);
% i8_r2 = uint8(randi([0 1],28,28,size(i8,3)));
% i9_m12 = uint8(randi([254 255],28,28,size(i9,3)));
% 
% images(:,:,find(labels==4))=i4_r0_m1;
% images(:,:,find(labels==5))=i5_r2_m1;
% images(:,:,find(labels==6))=i6_r3;
% images(:,:,find(labels==7))=i7_r1_m3;
% images(:,:,find(labels==8))=i8_r2;
% images(:,:,find(labels==9))=i9_m12;
% 
% images = permute(images,[2 1 3]);
% newStream=reshape(images,numel(images),1);

%%
%%%%%%%%%%%%%%% Replace 0 1 by random white and black images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate artificial 4 5 6 
% i0_r = uint8(randi([0 1],28,28,size(i0,3)));
% i1_r = uint8(randi([254 255],28,28,size(i1,3)));
% i2_r = uint8(randi([50 55],28,28,size(i2,3)));
% i3_r = uint8(randi([150 160],28,28,size(i3,3)));
% 
% images(:,:,find(labels==0))=i0_r;
% images(:,:,find(labels==1))=i1_r;
% images(:,:,find(labels==2))=i2_r;
% images(:,:,find(labels==3))=i3_r;
% 
% images = permute(images,[2 1 3]);
% newStream=reshape(images,numel(images),1);

%%
%%%%%%%%%%%%%%% Replace 2 3 by random distoted 0 1 images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate artificial 4 5 6 
i2_r = rotateDigitMNIST(size(i2,3),i1);
i3_r = rotateMergeDigitMNIST(size(i3,3),i0,i1,1);

images(:,:,find(labels==2))=i2_r;
images(:,:,find(labels==3))=i3_r;

images = permute(images,[2 1 3]);
newStream=reshape(images,numel(images),1);

%% write new data file
newfile= '/home/shaogang/caffe/data/mnist/train-images-replace-23-by-01';
fid=fopen(newfile,'wb');
fwrite(fid,magic1,'int32', 'ieee-be');
fwrite(fid,numImages,'int32', 'ieee-be');
fwrite(fid,numRows,'int32', 'ieee-be');
fwrite(fid,numCols,'int32', 'ieee-be');
fwrite(fid,newStream,'unsigned char');
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% read back to verify
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fp = fopen(newfile, 'rb');
% assert(fp ~= -1, ['Could not open ', newfile, '']);
% magic = fread(fp, 1, 'int32', 0, 'ieee-be');
% assert(magic == 2051, ['Bad magic number in ', newfile, '']);
% numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
% numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
% numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
% images = fread(fp, inf, 'unsigned char');
% images = reshape(images, numCols, numRows, numImages);
% images = permute(images,[2 1 3]);
% fclose(fp);
