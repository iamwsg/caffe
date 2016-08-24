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
for ii=601:700
    imgPath=cats{1}{ii};
    sp=strsplit(imgPath,'/');
    imgPath=strcat('/', sp{2},'/','shaogang','/',sp{4},'/',sp{5},'/',sp{6},'/',sp{7});
    disp(imgPath)
    
    ims1=Image_aug_color(imgPath,n_colors,resize);
    
    pos1 = ims1(:, :, [3, 2, 1],:); % convert from RGB to BGR
    pos1 = permute(pos1, [2, 1, 3, 4]); % permute width and height
    pos1 = single(pos1); % convert to single precision
   
    pos_size=size(pos1);
    n_positive=pos_size(4);
    
    net.blobs('data').reshape([resize resize 3 n_positive]); % reshape blob 'data'
    net.reshape();
    res_pos1 = net.forward({pos1});
    pos_feat1 = net.blobs('fc8').get_data()';

    %%store featues
    scells=strsplit(imgPath,'/');
    nsc=length(scells);
    fileName=['posFeats205/' scells{nsc-1} '_' scells{nsc} '.mat'];
    save(fileName, vname(pos_feat1));
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






