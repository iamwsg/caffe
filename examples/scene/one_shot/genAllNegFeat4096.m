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

%%
% files=dir('imageLists2');
% filesNew=dir('imageLists3');
% f=cell(length(files),1);
% for ii=1:length(files)
%     f{ii}=files(ii).name;
% end
% f2=cell(length(filesNew),1);
% for ii=1:length(filesNew)
%     f2{ii}=filesNew(ii).name;
% end
% df=setdiff(f2,f);
df=dir('/home/shaogang/Datasets/P205/');

%%
vname=@(x) inputname(1);
for ii=3:length(df)
    f=df(ii).name;
    fid=fopen(['/home/shaogang/Datasets/P205/' f]);
    tline = fgets(fid);
    feat=[];
    kk=1;
    while ischar(tline)
        disp(tline)
        %split=strsplit(tline,'~');
        %tline=['/home/shaogangwang' split{2}];
        tline=strtrim(tline);
        try 
            im_data=caffe.io.load_image(tline);
        catch
            disp('broken image')
            tline = fgets(fid);
            continue;
        end
        im_data = imresize(im_data, [resize, resize]);
        if ndims(im_data)~=3
            disp('outlier');
            tline = fgets(fid);
            continue;
        end
        net.forward({im_data});
        feat(kk,:)=net.blobs('fc7').get_data()';
        kk=kk+1;
        tline = fgets(fid);
    end
    fclose(fid);
    savefeat=['/home/shaogang/Datasets/negFeats4096/' f '.mat'];
    save(savefeat, vname(feat));
end












