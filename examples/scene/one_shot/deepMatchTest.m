%% Deep MatchNet test

clear;
close all;
clc;

addpath /home/shaogangwang/mywork/caffe/matlab
addpath /home/shaogangwang/mywork/caffe/examples/scene/one_shot

%% prepare the net
cd('/home/shaogangwang/mywork/caffe/');
disp('prepare net')
caffe.reset_all();
caffe.set_mode_gpu();
model = 'examples/scene/models/19_stream_train11_300000_MAXpool_conv3/matchNetTestHingeMini.prototxt';
weights = 'examples/scene/models/19_stream_train11_300000_MAXpool_conv3/scene_iter_35000.caffemodel';
%model = 'examples/scene/models/19_stream_train7_20000_MAXpool_pad_M/matchNetTestHingeMini.prototxt';
%weights = 'examples/scene/models/19_stream_train11_300000_MAXpool_conv3/scene_iter_35000.caffemodel';

net = caffe.Net(model, weights, 'test');
a=net.layers('InnerProduct2').params(1).get_data();
p=[];label=[];
for ii=1:10
    net.forward_prefilled();
    p = [p; squeeze(net.blobs('p').get_data())];
    label = [label; net.blobs('label').get_data()];
% pad = squeeze(net.blobs('pad').get_data());
% th = squeeze(net.blobs('th').get_data());
end

%%evaluation
nTest=length(label);
dth=-12:.1:5.4;
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=p;
    for jj=1:nTest
        if Res(jj)<dth(ii)
            Res(jj)=1;
        else
            Res(jj)=0;
        end
    end
    falseAlarm=find(label==1 & Res==0);
    pfa_dth(ii)=length(falseAlarm)/length(find(label==1));
    pd=find(label==0 & Res==0);
    pd_dth(ii)= length(pd)/length(find(label==0));
    ap_dth(ii)=length(find(Res==0 & label==0))/length(find(Res==0));
    recall_dth(ii)=length(find(label==0 & Res==0))/length(find(label==0));
end
figure,plot(pfa_dth,pd_dth,'-ob'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
%figure,plot(recall_dth,ap_dth,'-or'),grid;
%title('AP-RECALL'),xlabel('Recall'),ylabel('AP');
%legend('Dist thresholding','SVM thresholding');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
caffe.reset_all();
caffe.set_mode_gpu();
model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTestHingeMini_deploy.prototxt';
%model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTrainHingeMini.prototxt';
%weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_2000_loss_0.3.caffemodel';
weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_5000_loss_0.caffemodel';
net = caffe.Net(model, weights, 'test');

p=[];label=[];
for ii=1:10
    net.forward_prefilled();
    p = [p; squeeze(net.blobs('p').get_data())];
    label = [label; net.blobs('label').get_data()];
% pad = squeeze(net.blobs('pad').get_data());
% th = squeeze(net.blobs('th').get_data());
end

%%evaluation
nTest=length(label);
%dth=-1.05:.01:1;
dth=-12:.1:5.4;
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=p;
    for jj=1:nTest
        if Res(jj)<dth(ii)
            Res(jj)=1;
        else
            Res(jj)=0;
        end
    end
    falseAlarm=find(label==1 & Res==0);
    pfa_dth(ii)=length(falseAlarm)/length(find(label==1));
    pd=find(label==0 & Res==0);
    pd_dth(ii)= length(pd)/length(find(label==0));
    ap_dth(ii)=length(find(Res==0 & label==0))/length(find(Res==0));
    recall_dth(ii)=length(find(label==0 & Res==0))/length(find(label==0));
end
figure,plot(pfa_dth,pd_dth,'-ob'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
%figure,plot(recall_dth,ap_dth,'-or'),grid;

%%
caffe.reset_all();
caffe.set_mode_gpu();
model = 'examples/scene/models/baseLine_train7_20000/matchNetTestHingeMini.prototxt';
%model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTrainHingeMini.prototxt';
%weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_2000_loss_0.3.caffemodel';
weights = 'examples/scene/models/baseLine_train7_20000/_iter_5000.caffemodel';
net = caffe.Net(model, weights, 'test');

p=[];label=[];
for ii=1:10
    net.forward_prefilled();
    dt = net.blobs('dt').get_data();
    p=[p dt(1,:)];
    label = [label; net.blobs('label').get_data()];
% pad = squeeze(net.blobs('pad').get_data());
% th = squeeze(net.blobs('th').get_data());
end
label=label';

%%evaluation
nTest=length(label);
%dth=-1.05:.001:-0.96;
dth=-12:.1:8;
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=p;
    for jj=1:nTest
        if Res(jj)<dth(ii)
            Res(jj)=1;
        else
            Res(jj)=0;
        end
    end
    falseAlarm=find(label==1 & Res==0);
    pfa_dth(ii)=length(falseAlarm)/length(find(label==1));
    pd=find(label==0 & Res==0);
    pd_dth(ii)= length(pd)/length(find(label==0));
    ap_dth(ii)=length(find(Res==0 & label==0))/length(find(Res==0));
    recall_dth(ii)=length(find(label==0 & Res==0))/length(find(label==0));
end
figure,plot(pfa_dth,pd_dth,'-ob'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
baseLine_pfa=pfa_dth; baseLine_pd=pd_dth;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

caffe.reset_all();
caffe.set_mode_gpu();
model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTestHingeMini_deploy.prototxt';
%model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTrainHingeMini.prototxt';
%weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_2000_loss_0.3.caffemodel';
weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_5000_loss_0.caffemodel';
net = caffe.Net(model, weights, 'test');

p=[];label=[];
for ii=1:10
    net.forward_prefilled();
    p = [p; squeeze(net.blobs('p').get_data())];
    label = [label; net.blobs('label').get_data()];
% pad = squeeze(net.blobs('pad').get_data());
% th = squeeze(net.blobs('th').get_data());
end

%%evaluation
nTest=length(label);
%dth=-1.05:.01:1;
dth=-12:.1:5.4;
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=p;
    for jj=1:nTest
        if Res(jj)<dth(ii)
            Res(jj)=1;
        else
            Res(jj)=0;
        end
    end
    falseAlarm=find(label==1 & Res==0);
    pfa_dth(ii)=length(falseAlarm)/length(find(label==1));
    pd=find(label==0 & Res==0);
    pd_dth(ii)= length(pd)/length(find(label==0));
    ap_dth(ii)=length(find(Res==0 & label==0))/length(find(Res==0));
    recall_dth(ii)=length(find(label==0 & Res==0))/length(find(label==0));
end
figure,plot(pfa_dth,pd_dth,'-ob'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
%figure,plot(recall_dth,ap_dth,'-or'),grid;
s3_max_pfa=pfa_dth; s3_max_pd=pd_dth;

%%
caffe.reset_all();
caffe.set_mode_gpu();
model = 'examples/scene/models/3_stream_train7_20000_AVEpool/matchNetTestHingeMini.prototxt';
%model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTrainHingeMini.prototxt';
%weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_2000_loss_0.3.caffemodel';
weights = 'examples/scene/models/3_stream_train7_20000_AVEpool/scene_iter_5000.caffemodel';
net = caffe.Net(model, weights, 'test');

p=[];label=[];
for ii=1:10
    net.forward_prefilled();
    dt = squeeze(net.blobs('p').get_data());
    p=[p dt(1,:)];
    label = [label; net.blobs('label').get_data()];
% pad = squeeze(net.blobs('pad').get_data());
% th = squeeze(net.blobs('th').get_data());
end
label=label';

%%evaluation
nTest=length(label);
%dth=-1.05:.001:-0.96;
dth=-12:.1:8;
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=p;
    for jj=1:nTest
        if Res(jj)<dth(ii)
            Res(jj)=1;
        else
            Res(jj)=0;
        end
    end
    falseAlarm=find(label==1 & Res==0);
    pfa_dth(ii)=length(falseAlarm)/length(find(label==1));
    pd=find(label==0 & Res==0);
    pd_dth(ii)= length(pd)/length(find(label==0));
    ap_dth(ii)=length(find(Res==0 & label==0))/length(find(Res==0));
    recall_dth(ii)=length(find(label==0 & Res==0))/length(find(label==0));
end
figure,plot(pfa_dth,pd_dth,'-ob'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
s3_ave_pfa=pfa_dth; s3_ave_pd=pd_dth;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
caffe.reset_all();
caffe.set_mode_gpu();
model = 'examples/scene/models/19_stream_train7_20000_AVEpool/matchNetTestHingeMini.prototxt';
%model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTrainHingeMini.prototxt';
%weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_2000_loss_0.3.caffemodel';
weights = 'examples/scene/models/19_stream_train7_20000_AVEpool/scene_iter_10000.caffemodel';
net = caffe.Net(model, weights, 'test');

p=[];label=[];
for ii=1:10
    net.forward_prefilled();
    dt = squeeze(net.blobs('p').get_data());
    p=[p dt(1,:)];
    label = [label; net.blobs('label').get_data()];
% pad = squeeze(net.blobs('pad').get_data());
% th = squeeze(net.blobs('th').get_data());
end
label=label';

%%evaluation
nTest=length(label);
%dth=-1.05:.001:-0.96;
dth=-12:.1:8;
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=p;
    for jj=1:nTest
        if Res(jj)<dth(ii)
            Res(jj)=1;
        else
            Res(jj)=0;
        end
    end
    falseAlarm=find(label==1 & Res==0);
    pfa_dth(ii)=length(falseAlarm)/length(find(label==1));
    pd=find(label==0 & Res==0);
    pd_dth(ii)= length(pd)/length(find(label==0));
    ap_dth(ii)=length(find(Res==0 & label==0))/length(find(Res==0));
    recall_dth(ii)=length(find(label==0 & Res==0))/length(find(label==0));
end
figure,plot(pfa_dth,pd_dth,'-ob'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
s19_ave_pfa=pfa_dth; s19_ave_pd=pd_dth;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
caffe.reset_all();
caffe.set_mode_gpu();
model = 'examples/scene/models/19_stream_train7_20000_MAXpool/matchNetTestHingeMini.prototxt';
%model = 'examples/scene/models/3_stream_train7_20000_pad/matchNetTrainHingeMini.prototxt';
%weights = 'examples/scene/models/3_stream_train7_20000_pad/scene_iter_2000_loss_0.3.caffemodel';
weights = 'examples/scene/models/19_stream_train7_20000_MAXpool/scene_iter_10000.caffemodel';
net = caffe.Net(model, weights, 'test');

p=[];label=[];
for ii=1:10
    net.forward_prefilled();
    dt = squeeze(net.blobs('p').get_data());
    p=[p; dt];
    label = [label; net.blobs('label').get_data()];
% pad = squeeze(net.blobs('pad').get_data());
% th = squeeze(net.blobs('th').get_data());
end


%%evaluation
nTest=length(label);
%dth=-1.05:.001:-0.96;
dth=-12:.1:8;
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=p;
    for jj=1:nTest
        if Res(jj)<dth(ii)
            Res(jj)=1;
        else
            Res(jj)=0;
        end
    end
    falseAlarm=find(label==1 & Res==0);
    pfa_dth(ii)=length(falseAlarm)/length(find(label==1));
    pd=find(label==0 & Res==0);
    pd_dth(ii)= length(pd)/length(find(label==0));
    ap_dth(ii)=length(find(Res==0 & label==0))/length(find(Res==0));
    recall_dth(ii)=length(find(label==0 & Res==0))/length(find(label==0));
end
figure,plot(pfa_dth,pd_dth,'-ob'),grid;
title('ROC'),xlabel('P_{fa}'),ylabel('P_d');
s19_max_pfa=pfa_dth; s19_max_pd=pd_dth;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
figure, plot(baseLine_pfa, baseLine_pd, '-ob', s3_ave_pfa, s3_ave_pd, '-og', s3_max_pfa, s3_max_pd, '-oblack', s19_ave_pfa, s19_ave_pd, '-*r', s19_max_pfa, s19_max_pd,'-*m'),grid;
title('ROC');xlabel('P_{fa}');ylabel('P_d');
legend('SiameseNet','3 Streams AVE pool','3 Streams MAX pool','19 Streams AVE pool','19 Streams MAX pool');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot(1:100,label,1:100,sp,'r'),grid
%% visialize input images
% i1=net.blobs('i1').get_data();
% i2=net.blobs('i2').get_data();
% i11=i1(:,:,:,2);
% i12=i2(:,:,:,2);
% i11 = permute(i11, [2, 1, 3, 4]);
% i12 = permute(i12, [2, 1, 3, 4]);
% i11 = i11(:, :, [3, 2, 1],:);
% i12 = i12(:, :, [3, 2, 1],:);
% figure, subplot(211),imshow(i11),subplot(212),imshow(i12);







