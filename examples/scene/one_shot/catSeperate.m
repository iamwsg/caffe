%% store each image category into seperate files
fileId=fopen('newImageList.txt');
cats=textscan(fileId,'%s');
fclose(fileId);

n=length(cats{1});

firstLine=cats{1}(1);
firstCells=strsplit(firstLine{1},'/');
firstCat=firstCells{(length(firstCells)-1)};
lastCat=firstCat;
sep=[];
for ii=2:n
    s=cats{1}(ii);
    cells=strsplit(s{1},'/');
    cat=cells{(length(cells)-1)};
    
    if strcmp(cat,'outdoor')
        cat=[cells{(length(cells)-2)},'_outdoor'];
    end
    
    if ~strcmp(cat,lastCat)
        sep=[sep ii];
        lastCat=cat;
    end
    
end

sep=[1 sep n];

for ii=1:length(sep)-1
    %get the filename 
    firstLine=cats{1}(sep(ii));
    firstCells=strsplit(firstLine{1},'/');
    firstCat=firstCells{(length(firstCells)-1)};
    if strcmp(firstCat,'outdoor')
        firstCat=[firstCells{(length(firstCells)-2)},'_outdoor'];
    end
    fileName=['imageLists3/' firstCat '.txt'];
    fid=fopen(fileName, 'w');
    for jj=sep(ii):sep(ii+1)-1
        str=cats{1}(jj);
        fprintf(fid,'%s\n',str{1});
    end
    fclose(fid);
end