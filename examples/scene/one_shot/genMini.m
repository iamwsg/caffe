%% gen all positive features
%%gen all negative features
clear;
close all;
clc;

%%prepare files
fileId=fopen('sameScene.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

n=length(cats{1});

resize=32;

vname=@(x) inputname(1);

%%
for ii=1:n
    imgPath=cats{1}{ii};
    disp(imgPath)
    
    ims1=imread(imgPath);
    ims1=imresize(ims1,[resize,resize]);
    pos_feat=reshape(ims1,[1,resize*resize*3]);

    %%store featues
    scells=strsplit(imgPath,'/');
    nsc=length(scells);
    fileName=['posFeatsMini/' scells{nsc-1} '_' scells{nsc} '.mat'];
    save(fileName, vname(pos_feat));
end


