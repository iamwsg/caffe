%% gen all positive features
%%gen all negative features
clear;
close all;
clc;

addpath /home/shaogangwang/mywork/caffe/matlab
addpath /home/shaogangwang/mywork/caffe/examples/scene/one_shot

%prepare the net
caffe.reset_all();
caffe.set_mode_gpu();
model = '/home/shaogangwang/Downloads/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/home/shaogangwang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
net = caffe.Net(model, weights, 'test');
resize=227;
net.blobs('data').reshape([resize resize 3 10]); % reshape blob 'data'
net.reshape();

%%prepare files
fileId=fopen('sameScene.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

n=length(cats{1});
n_colors=1;
vname=@(x) inputname(1);
for ii=1:n
    imgPath=cats{1}{ii};
    disp(ii)
    disp(imgPath)
    ims1=denseImages(imgPath);
    pos1 = ims1(:, :, [3, 2, 1],:); % convert from RGB to BGR
    pos1 = permute(pos1, [2, 1, 3, 4]); % permute width and height
    pos1 = single(pos1); % convert to single precision
    pos_size=size(pos1);
    n_positive=pos_size(4);
%     resize=227;
%     net.blobs('data').reshape([resize resize 3 n_positive]); % reshape blob 'data'
%     net.reshape();
    net.forward({pos1});
    pos_feat1 = net.blobs('fc7').get_data()';
    
    %%store featues
    scells=strsplit(imgPath,'/');
    nsc=length(scells);
    fileName=['posFeatsDense/' scells{nsc-1} '_' scells{nsc} '.mat'];
    save(fileName, vname(pos_feat1));
end
