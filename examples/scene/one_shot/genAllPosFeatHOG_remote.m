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
n_colors=1;
resize=227;
blk=16;
vname=@(x) inputname(1);

%%

for ii=1:n
    imgPath=cats{1}{ii};
    disp(imgPath)
    cpath=strsplit(imgPath,'/');
    imgPath=strcat('/home/shaogang/',cpath{4},'/',cpath{5},'/',cpath{6},'/',cpath{7});
    
    
    ims1=Image_aug_color(imgPath,n_colors,resize);
        
    [~,~,~,n_positive]=size(ims1);
    
    pos_feat=zeros(n_positive,9216);
    for ii=1:n_positive
        pos_feat(ii,:) = extractHOGFeatures(ims1(:,:,:,ii),'BlockSize',[blk blk]);
        
    end

    %%store featues
    scells=strsplit(imgPath,'/');
    nsc=length(scells);
    fileName=['/home/shaogang/Datasets/posFeatsHOG/' scells{nsc-1} '_' scells{nsc} '.mat'];
    save(fileName, vname(pos_feat));
end