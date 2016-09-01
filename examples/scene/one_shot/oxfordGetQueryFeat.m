%% get query feat
clear;
close all;

% find query files
root='/home/shaogang/Downloads/gt_files_170407/';
imageRoot='/home/shaogang/Downloads/oxford/oxbuild_images/';
files=dir(root);

%prepare the net
caffe.reset_all();
caffe.set_mode_cpu();
model = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt';
weights = '/home/shaogang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel';
net = caffe.Net(model, weights, 'test');
resize=227;
net.blobs('data').reshape([resize resize 3 1]); % reshape blob 'data'
net.reshape();

n=55;
query205=single(zeros(n,205));
queryProb=single(zeros(n,205));
query4096=single(zeros(n,4096));
name=cell(n,1);

for ii=1:n
    disp(ii)
    f=files(ii*4+2).name;
    fpath=strcat(root,f);
    fid=fopen(fpath);
    query=textscan(fid,'%s');
    fclose(fid);
    q=query{1}{1};
    q=q(6:end);
    
    imgPath=strcat(imageRoot,q,'.jpg');
    im=imread(imgPath);
    x1=str2num(query{1}{2});
    y1=str2num(query{1}{3});
    x2=str2num(query{1}{4});
    y2=str2num(query{1}{5});
    rec=[x1,y1,x2-x1,y2-y1];
    crop=imcrop(im,rec);
    ims1=imresize(crop,[227,227]);
    
    pos1 = ims1(:, :, [3, 2, 1]); % convert from RGB to BGR
    pos1 = permute(pos1, [2, 1, 3]); % permute width and height
    pos1 = single(pos1); % convert to single precision
    
    net.forward({pos1});
    query205(ii,:) = net.blobs('fc8').get_data()';
    queryProb(ii,:) = net.blobs('prob').get_data()';
    query4096(ii,:) = net.blobs('fc7').get_data()';
    name{ii}=q;
end

oxfordQuery.name=name;
oxfordQuery.query205=query205;
oxfordQuery.query4096=query4096;
oxfordQuery.queryProb=queryProb;

