%%select top categories
clear;
close all;
clc;

addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

txtPath='/media/sf_Datasets/Scenes/imageList.txt';
fileID = fopen(txtPath);
N = textscan(fileID,'%s %d');
fclose(fileID);

tic
disp('prepare net')
%%create net
caffe.reset_all();
model = '/media/sf_Datasets/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/media/sf_Datasets/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
labelFile='/media/sf_Datasets/placesCNN_upgraded/label205.csv';
resize=227;

caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');
toc
%%labels
fileID = fopen(labelFile);
C = textscan(fileID,'%s %d');
fclose(fileID);

%% resize Network
net.blobs('data').reshape([resize resize 3 1]); % reshape blob 'data'
net.reshape();
%% coarser class
n_neg = length(N{1});

cats=zeros(n_neg*5,1);
for ii=1:n_neg
    im_path=strcat('/media/sf_Datasets/Scenes/', N{1}(ii));
    %figure,imshow(imread(im_path{1}));
    im=caffe.io.load_image(im_path{1});
    im = imresize(im, [resize, resize]);
    if ndims(im)~=3
        disp('outlier');
        disp(im_path);
        continue;
    end
    res = net.forward({im});
    r1=res{1};
    n=5;
    [sortedX,sortingIndices] = sort(r1,'descend');
    maxValues = sortedX(1:n);
    maxValueIndices = sortingIndices(1:n);
    cats(n*(ii-1)+1:n*ii)=maxValueIndices;
    disp(ii);
end

ucats=unique(cats);
sucats=sort(ucats);
topCats=C{1}(sucats);

fileID = fopen('topCats.txt','w');
for ii=1:length(topCats)
    fprintf(fileID, '%s\n',topCats{ii});
end
fclose(fileID);


