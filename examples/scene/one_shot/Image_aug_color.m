function imss=Image_aug_color(image_path,n_color,resize)

im_origin = imread(image_path);
ims=Imag_aug(im_origin,resize);
s=size(ims);
imss=uint8(zeros(s(1),s(2),s(3),s(4)*(n_color+1)));
N=s(4);
imss(:,:,:,1:N)=ims;

for ii=1:n_color
    RGB2 = imadjust(im_origin,[0.5*rand(1,3); 0.5+0.5*rand(1,3)],[]);
    ims=Imag_aug(RGB2,resize);
    imss(:,:,:,ii*N+1:(ii+1)*N)=ims;
end
