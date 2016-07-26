%% Image augumentation
function ims=Imag_aug(im_origin,s)
% rotation
%image_path='/media/sf_Datasets/Scenes/images/Taj_Mahal/32.jpg';
%im_origin = imread(image_path);
%figure,imshow(im);

%s=256;
N=125;
ims=uint8(zeros(s,s,3,N));

%resize to dds*dds
dds=640;
im = imresize(im_origin,[dds,dds]);
%figure,imshow(im);

% rotate and crop
ds=512;
im_crop=uint8(zeros(ds,ds,3,5));
for ii=1:5
    im_rot = imrotate(im,-6+(ii-1)*3);
    si=size(im_rot);
    x_min = round((si(1)-ds)/2);
    im_crop(:,:,:,ii)=imcrop(im_rot, [x_min,x_min,ds-1,ds-1]);
    %subplot(211),imshow(im_rot),subplot(212),imshow(im_crop(:,:,:,ii))
end

for ii=1:5
    im_re=imresize(im_crop(:,:,:,ii),[512,256]);
    im_c{(ii-1)*5+1}=im_re;
    im_re=imresize(im_crop(:,:,:,ii),[512,384]);
    im_c{(ii-1)*5+2}=im_re;
    im_c{(ii-1)*5+3}=im_crop(:,:,:,ii);
    im_re=imresize(im_crop(:,:,:,ii),[384,512]);
    im_c{(ii-1)*5+4}=im_re;
    im_re=imresize(im_crop(:,:,:,ii),[256,512]);
    im_c{(ii-1)*5+5}=im_re;
end

% for ii=1:25
%     imshow(im_c{ii})
%     pause(1);
% end

% random crop 
for ii=1:25
    img=im_c{ii};
    img_size=size(img);
    wid=img_size(1)-s;
    high=img_size(2)-s;
    
    for jj=1:5
        x_min=randi(high+1);
        y_min=randi(wid+1);
        ims(:,:,:,(ii-1)*5+jj)=imcrop(img,[x_min,y_min,s-1,s-1]);
    end
end

% for ii=1:125
%     subplot(211),imshow(im_origin);
%     subplot(212),imshow(ims(:,:,:,ii));
%     pause(1);
% end

%% adjust color 
% n_color=2;
% imss=uint8(zeros(s,s,3,N*(n_color+1)));
% imss(:,:,:,1:N)=ims;
% 
% for ii=1:n_color
%     
% end
% 
% image_path='/media/sf_Datasets/Scenes/images/Taj_Mahal/32.jpg';
% im_origin = imread(image_path);
% for ii=1:5
%     RGB2 = imadjust(im_origin,[0.5*rand(1,3); 0.5+0.5*rand(1,3)],[]);
%     subplot(211),imshow(im_origin);
%     subplot(212),imshow(RGB2);
%     pause(1);
% end


