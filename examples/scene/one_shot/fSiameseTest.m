%% test 789
%% create net
function [pfa_dth,pd_dth,recall_dth,ap_dth]=fSiameseTest(model, weights)

caffe.reset_all();
%model = '/home/shaogang/caffe/examples/siamese/mnist_siamese_train_test_sim_deploy.prototxt';
%weights = '/home/shaogang/caffe/examples/siamese/My_mnist_siamese_0to9_iter_5000.caffemodel';
%weights = '/home/shaogang/caffe/examples/siamese/mnist_siamese_0to6_feat_2_iter_3000.caffemodel';
%weights = '/home/shaogang/caffe/examples/siamese/My_mnist_siamese_0to3_t89_feat_2_sim_iter_3000.caffemodel';

caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');
net.forward_prefilled();
prob = net.blobs('fc3').get_data();
label = net.blobs('label').get_data();
acc = net.blobs('accuracy').get_data();
%%
comp(1,:)=prob(1,:);
comp(2,:)=label;
comp(1,find(prob(1,:)<0))=0;
comp(1,find(prob(1,:)>=0))=1;
eq=find(comp(1,:)==comp(2,:));
%%
soft=softmax(prob);
dists=soft(1,:);
nTest=length(label);
tRes=cell(nTest, 15);
tRes(:,3)=num2cell(label);
tRes(:,4)=num2cell(dists);

dth=linspace(min(dists),max(dists)); 
ndth=length(dth);
pfa_dth=zeros(1,ndth);
pd_dth=zeros(1,ndth);
ap_dth=pd_dth;
recall_dth=pd_dth;
for ii=1:ndth
    Res=tRes;
    for jj=1:nTest
        if Res{jj,4}>dth(ii)
            Res{jj,15}=0;
        else
            Res{jj,15}=1;
        end
    end
    falseAlarm=find(([Res{:,3}]==1) & ([Res{:,15}]==0));
    pfa_dth(ii)=length(falseAlarm)/length(find(([Res{:,3}]==1)));
    pd=find(([Res{:,3}]==0) & ([Res{:,15}]==0));
    pd_dth(ii)= length(pd)/length(find([Res{:,3}]==0));
    ap_dth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,15}]==0));
    recall_dth(ii)=length(find(([Res{:,3}]==0) & ([Res{:,15}]==0)))/length(find([Res{:,3}]==0));
end

end