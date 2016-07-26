function fileList = filePathGenNoNum(dirName,fileName)

  dirData = dir(dirName);      %# Get the data for the current directory
  dirData=dirData(~ismember({dirData.name},{'.','..','.DS_Store'}));
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
                                               
  fileID = fopen(fileName,'w');
                                               
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = getAllFiles(nextDir);
    for ii=1:length(fileList)
        fprintf(fileID,'%s %d\n',strjoin(fileList(ii)));
    end
 
         
    %fprintf(fileID,'%6s %12s\n','x','exp(x)');
    %fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end

  fclose(fileID);
end