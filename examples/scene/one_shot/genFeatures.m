%%scene matching
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
net.blobs('data').reshape([resize resize 3 1]); % reshape blob 'data'
net.reshape();

%%
files=dir('imageLists2');
vname=@(x) inputname(1);
for ii=77:77 %length(files)
    f=files(ii).name;
    
    fid=fopen(['imageLists2/' f]);
    tline = fgets(fid);
    feat=[];
    kk=1;
    while ischar(tline)
        disp(tline)
        split=strsplit(tline,'~');
        tline=['/home/shaogangwang' split{2}];
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
    savefeat=['negFeats/' f '.mat'];
    save(savefeat, vname(feat));
end