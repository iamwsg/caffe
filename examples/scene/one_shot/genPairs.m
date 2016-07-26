%% This script generate image pairs 
%% input: images.txt
%% output: pairs.txt
clear;

%read image files and labels, and store in cell vectors
fid_read = fopen('test_unseen.txt');
fid = fopen('test_pairs_unseen.txt','w');
Np=1000;
tline = fgets(fid_read);
lines={tline};
ii=2;
while ischar(tline)
    disp(tline)
    tline = fgets(fid_read);
    lines(ii)={tline};
    ii=ii+1;
end
fclose(fid_read);

s=strsplit(char(lines(1)),' ')

N=length(lines);

for ii=1:N-1
    split(ii)={strsplit(char(lines(ii)),' ')};
end

%%Generate pairs

R=randi(N-1,Np,2);
for ii=1:Np
    s1=split(R(ii,1));
    s2=split(R(ii,2));
    if s1{1,1}{1,2} == s2{1,1}{1,2}
        label = '0';
    else
        label = '1';
    end
    newline = [s1{1,1}{1,1}, ' ',s2{1,1}{1,1}, ' ',label];
    fprintf(fid,'%s\n',newline);
end
fclose(fid);


