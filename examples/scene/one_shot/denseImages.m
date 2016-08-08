%% dense compare deep features
function ims = denseImages(imagPath)
%imagPath='';
im_origin=imread(imagPath);
resize=227;
im_p=imresize(im_origin, [resize, resize]);

lg=resize*2;
im_large=imresize(im_origin, [lg,lg]);
st=floor(resize/2);
st2=2*st;

ims=uint8(zeros(resize,resize,3,10));
resize=resize-1;
ims(:,:,:,1)=im_p;
ims(:,:,:,2)= imcrop(im_large,[1,1,resize,resize]);
ims(:,:,:,3)= imcrop(im_large,[st,1,resize,resize]);
ims(:,:,:,4)= imcrop(im_large,[st2,1,resize,resize]);
ims(:,:,:,5)= imcrop(im_large,[1,st,resize,resize]);
ims(:,:,:,6)= imcrop(im_large,[st,st,resize,resize]);
ims(:,:,:,7)= imcrop(im_large,[st2,st,resize,resize]);
ims(:,:,:,8)= imcrop(im_large,[1,st2,resize,resize]);
ims(:,:,:,9)= imcrop(im_large,[st,st2,resize,resize]);
ims(:,:,:,10)= imcrop(im_large,[st2,st2,resize,resize]);