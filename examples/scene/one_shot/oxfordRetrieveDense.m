%% scenen retrival
clear;
close all;
clc;
addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

root='/home/shaogang/Downloads/gt_files_170407/';
files=dir(root);

%%prepare files
%fileId=fopen('sameScene.txt');
fileId=fopen('oxford.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

featRoot='oxfordFeatsDense205/';
featFile=dir(featRoot);
n=length(featFile)-2;
dfeats=single(zeros(n,10,205));
for ii=1:n
    load(strcat(featRoot,featFile(ii+2).name));
    dfeats(ii,:,:)=pos_feat1;
end

%%
%feat=load('oxford.mat');
%feat=feat.oxford.posProb;

%% 
load('oxfordQuery.mat');
query=oxfordQuery.query205;

%% load the whiter
% w=load('mau_whiter.mat');
% w=w.whiter;

vname=@(x) inputname(1);

%%
for jj=1:size(query,1)
    disp(jj);
   
    %% prepare test result containner
    tRes=cell(n, 5); %14:naive thresholding label; 15:two-step label; 16: first step mark
    
    %% the positive set
    f=files(jj*4-1).name;
    fpath=strcat(root,f);
    fid=fopen(fpath);
    pos1=textscan(fid,'%s');
    fclose(fid);
    
    f=files(jj*4+1).name;
    fpath=strcat(root,f);
    fid=fopen(fpath);
    pos2=textscan(fid,'%s');
    fclose(fid);
    pos=union(pos1{1,:},pos2{1,:});
    %% the junk set
    f=files(jj*4).name;
    fpath=strcat(root,f);
    fid=fopen(fpath);
    junk=textscan(fid,'%s');
    junk=junk{1};
    fclose(fid);
%% do the test
    for ii=1:n
        
        %disp(ii)
        
        tRes{ii,1}=oxfordQuery.name{jj};
        cat=cats{1}{ii};
        sp=strsplit(cat,'/');
        sp=strsplit(sp{7},'.');
        tRes{ii,2}=sp{1};
        
        if ismember(sp{1},pos)
            tRes{ii,3}=0;
        elseif ismember(sp{1},junk)
            tRes{ii,3}=2;
        else
            tRes{ii,3}=1;
        end
%         if strcmp(testScene,qScene)
%             tRes{ii,3}=0;
%         else 
%             tRes{ii,3}=1;
%         end

        r1=query(jj,:);
        r2=squeeze(dfeats(ii,:,:));
        
        %r1=(w*r1')'/length(r1);
        %r2=(w*r2')'/length(r1);
        
        %dist = r1*r2'/norm(r1)/norm(r2);
        [dist, in]= max(r1*r2');
        
        %dist = smax(r1)*smax(r2)';
        %dist=norm(w*r1'-w*r2');
        %dist = -smax(r1)*log2(smax(r2)')-smax(r2)*log2(smax(r1)');
        
        tRes{ii,4}=dist;
        tRes{ii,5}=in;
                
    end
    
    fileName=strcat('oxfordRetreveAngle205Densefast/oxfordRetreve_',oxfordQuery.name{jj},'.mat');
    save(fileName, vname(tRes));
end
