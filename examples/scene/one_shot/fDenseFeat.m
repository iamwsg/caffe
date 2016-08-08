function [minDis,in]= fDenseFeat(net, impath1, impath2)

ims1=denseImages(impath1);
ims2=denseImages(impath2);

pos1 = ims1(:, :, [3, 2, 1],:); % convert from RGB to BGR
pos1 = permute(pos1, [2, 1, 3, 4]); % permute width and height
pos1 = single(pos1); % convert to single precision
pos2 = ims2(:, :, [3, 2, 1],:); % convert from RGB to BGR
pos2 = permute(pos2, [2, 1, 3, 4]); % permute width and height
pos2 = single(pos2); % convert to single precision

pos_size=size(pos1);
n_positive=pos_size(4);

%tic
net.blobs('data').reshape([resize resize 3 n_positive]); % reshape blob 'data'
net.reshape();
net.forward({pos1});
pos_feat1 = net.blobs('fc7').get_data()';
net.forward({pos2});
pos_feat2 = net.blobs('fc7').get_data()';

dist=@(r1,r2)sqrt((r1-r2)*(r1-r2)')/norm(r1)/norm(r2);

%%do compare
dis=zeros(1,2*n_positive-1);
for ii=1:n_positive
    dis(ii)=dist(pos_feat1(1,:),pos_feat2(ii,:));
end
for ii=1:n_positive-1
    dis(n_positive+ii)=dist(pos_feat2(1,:),pos_feat1(ii+1,:));
end

[minDis, in]=min(dis)

% if in<=n_positive
%     figure,subplot(211),imshow(ims1(:,:,:,1)),subplot(212),imshow(ims2(:,:,:,in));
% else
%     figure,subplot(211),imshow(ims2(:,:,:,1)),subplot(212),imshow(ims1(:,:,:,in-n_positive));
% end