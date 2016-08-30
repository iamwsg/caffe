%% gen all positive features
%%gen all negative features
clear;
close all;
clc;

addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

%prepare the net
caffe.reset_all();
caffe.set_mode_cpu();
model = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
net = caffe.Net(model, weights, 'test');
resize=227;
net.blobs('data').reshape([resize resize 3 1]); % reshape blob 'data'
net.reshape();

%%prepare files
fileId=fopen('sameScene.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

n=length(cats{1});
n_colors=1;
vname=@(x) inputname(1);
pos205=single(zeros(700,205));
pos4096=single(zeros(700,4096));
posProb=single(zeros(700,205));
for ii=1:700
    imgPath=cats{1}{ii};
    sp=strsplit(imgPath,'/');
    imgPath=strcat('/', sp{2},'/','shaogang','/',sp{4},'/',sp{5},'/',sp{6},'/',sp{7});
    disp(imgPath)
    
    im_data1=caffe.io.load_image(imgPath);
    im_data1 = imresize(im_data1, [227, 227]);

    res_pos1 = net.forward({im_data1});
    pos205(ii,:) = net.blobs('fc8').get_data()';
    pos4096(ii,:) = net.blobs('fc7').get_data()';
    posProb(ii,:)=net.blobs('prob').get_data()';
    %%store featues
end










% n=length(cats{1});
% 
% firstLine=cats{1}(1);
% firstCells=strsplit(firstLine{1},'/');
% firstCat=firstCells{(length(firstCells)-1)};
% lastCat=firstCat;
% sep=[];
% for ii=2:n
%     s=cats{1}(ii);
%     cells=strsplit(s{1},'/');
%     cat=cells{(length(cells)-1)};
%     
%     if strcmp(cat,'outdoor')
%         cat=[cells{(length(cells)-2)},'_outdoor'];
%     end
%     
%     if ~strcmp(cat,lastCat)
%         sep=[sep ii];
%         lastCat=cat;
%     end
%     
% end
% 
% sep=[1 sep n];
% 
% %%
% vname=@(x) inputname(1);
% for ii=1:length(sep)-1
%     %get the filename 
%     firstLine=cats{1}(sep(ii));
%     firstCells=strsplit(firstLine{1},'/');
%     firstCat=firstCells{(length(firstCells)-1)};
%     if strcmp(firstCat,'outdoor')
%         firstCat=[firstCells{(length(firstCells)-2)},'_outdoor'];
%     end
%     fileName=['posFeats/' firstCat '.mat'];
%     
%     for jj=sep(ii):sep(ii+1)-1
%       
%     end
%     
% end






