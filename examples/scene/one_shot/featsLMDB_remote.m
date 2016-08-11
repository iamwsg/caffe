clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot
addpath /home/shaogang/matlab-lmdb

%% prepare file
files=dir('posFeatsDense');
filesTrain=files(3:552);
filesTest=files(553:702);

%% prepare DB
database = lmdb.DB('/home/shaogang/Datasets/FeatsDB/featsTrain20k','MAPSIZE', 8*1024^3);
%database = lmdb.DB('./db');
%% create
N=20000;
files=filesTrain;
len=length(files);


for ii=1:N
    disp(ii)
    ij=randi(len,1,2);
    file1=files(ij(1)).name;
    file2=files(ij(2)).name;
    
    cfile1=strsplit(file1,'_');
    cfile2=strsplit(file2,'_');
    if strcmp(cfile1{1},cfile2{1})
        label=-1;
    else
        label=1;
    end
    
    feat1=load(['posFeatsDense/',file1]);
    feat2=load(['posFeatsDense/',file2]);
    datum = caffe_pb.toDatum([feat1.pos_feat1;feat2.pos_feat1], label);
    key=int2str(ii);
    transaction = database.begin();
    try
      transaction.put(key,datum);
      transaction.commit();
    catch exception
        disp('transanction error')
      transaction.abort();
    end
end
