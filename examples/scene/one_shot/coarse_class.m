function [p1,p2,r1,r2,feat]=  coarse_class(net,C,size,image1,image2)

%im1=imresize(image1, 'OutputSize',[size, size]);
%im2=imresize(image2, 'OutputSize',[size, size]);
im_data1=caffe.io.load_image(image1);
im_data2=caffe.io.load_image(image2);
im_data1 = imresize(im_data1, [size, size]);
im_data2 = imresize(im_data2, [size, size]);

im=zeros(size,size,3,2);
im(:,:,:,1)=im_data1;
im(:,:,:,2)=im_data2;
res = net.forward({im});
r1=res{1}(:,1);
r2=res{1}(:,2);
feat=net.blobs('fc7').get_data()';
[M1,I1]=max(r1);
[M2,I2]=max(r2);
%stem(r),grid;

N=5;
[sortedX,sortingIndices] = sort(r1,'descend');
maxValues = sortedX(1:N);
maxValueIndices = sortingIndices(1:N);
for ii=1:N
    p1{ii}=[num2str(maxValues(ii)) ' ' C{1}{maxValueIndices(ii)} ' ' num2str(C{2}(maxValueIndices(ii)))];
end

[sortedX,sortingIndices] = sort(r2,'descend');
maxValues = sortedX(1:N);
maxValueIndices = sortingIndices(1:N);
for ii=1:N
    p2{ii}=[num2str(maxValues(ii)) ' ' C{1}{maxValueIndices(ii)} ' ' num2str(C{2}(maxValueIndices(ii)))];
end

