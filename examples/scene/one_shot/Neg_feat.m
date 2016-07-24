%% generate negtive feature set

function neg_feat = Neg_feat(negative_file_path, net, resize)

% negative_file_path='/media/sf_Datasets/Scenes/Taj_Mahal_negative.txt';
fileID = fopen(negative_file_path);
N = textscan(fileID,'%s %d');
fclose(fileID); 
%% resize Network
net.blobs('data').reshape([resize resize 3 1]); % reshape blob 'data'
net.reshape();
%% Calculate negative features
n_neg = length(N{1});
neg_feat=zeros(n_neg,4096);
for ii=1:n_neg
    im_path=strcat('/media/sf_Datasets/Scenes/', N{1}(ii));
    %figure,imshow(imread(im_path{1}));
    im=caffe.io.load_image(im_path{1});
    im = imresize(im, [resize, resize]);
    res = net.forward({im});
    neg_feat(ii,:) = net.blobs('fc7').get_data()';
    disp(ii);
end