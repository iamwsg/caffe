%% Get all negative HOG features
clear;

files=dir('imageLists3');
f=cell(length(files),1);
for ii=1:length(files)
    f{ii}=files(ii).name;
end


%%
blk=16;
resize=227;
vname=@(x) inputname(1);
for ii=4:length(f)
    fid=fopen(['imageLists3/' f{ii}]);
    tline = fgets(fid);
    feat=[];
    kk=1;
    while ischar(tline)
        disp(tline)
        %split=strsplit(tline,'~');
        %tline=['/home/shaogangwang' split{2}];
        tline=strtrim(tline);
        try 
            im_data=imread(tline);
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
        
        feat(kk,:) = extractHOGFeatures(im_data,'BlockSize',[blk blk]);
        
        
        kk=kk+1;
        tline = fgets(fid);
    end
    fclose(fid);
    savefeat=['negFeatHOG/' f{ii} '.mat'];
    save(savefeat, vname(feat));
end




