clear;
close all;
clc;

addpath /home/shaogang/caffe/matlab
addpath /home/shaogang/caffe/examples/scene/one_shot

model = '/home/shaogang/caffe/examples/siamese/mnist_siamese_train_test_sim_deploy.prototxt';
weights_09 = '/home/shaogang/caffe/examples/siamese/mnist_siamese_0to9_feat_2_fine_iter_6000.caffemodel';
weights_06 = '/home/shaogang/caffe/examples/siamese/mnist_siamese_0to6_feat_2_iter_3000.caffemodel';
weights_03 = '/home/shaogang/caffe/examples/siamese/My_mnist_siamese_0to3_t89_feat_2_sim_iter_3000.caffemodel';
weights_03r = '/home/shaogang/caffe/examples/siamese/mnist_siamese_0to3_replace_4to9_rotateMerge_feat_2_fine_iter_20000.caffemodel';
weights_01 = '/home/shaogang/caffe/examples/siamese/mnist_siamese_0to1_feat_2_iter_2000.caffemodel';


[pfa_09,pd_09,~,~]=fSiameseTest(model,weights_09);
[pfa_06,pd_06,~,~]=fSiameseTest(model,weights_06);
[pfa_03,pd_03,~,~]=fSiameseTest(model,weights_03);
[pfa_01,pd_01,~,~]=fSiameseTest(model,weights_01);
[pfa_03r,pd_03r,~,~]=fSiameseTest(model,weights_03r);

plot(pfa_01,pd_01,'b',pfa_03,pd_03,'r',pfa_06,pd_06,'black',pfa_03r,pd_03r,'m',...
    pfa_09,pd_09,'g','LineWidth',2),grid,xlabel('P_{fa}'),ylabel('P_d');
legend('01','03','06','03c','09');

