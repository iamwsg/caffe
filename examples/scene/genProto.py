caffe_root = '/home/shaogang/caffe/'  
folder=caffe_root+'/examples/scene/'
import sys
import os
sys.path.insert(0, caffe_root + 'python')
import caffe

from caffe import layers as L, params as P

trainBatchSize=128
testBatchSize=100
cropSize=64

def maxPool(bottom):
	top=L.Pooling(bottom, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	return top
    
n = caffe.NetSpec()

n.pair_data, n.label = L.Data(name = 'pair_data',batch_size=trainBatchSize, backend=P.Data.LMDB, source="examples/scene/train_pairs.lmdb", include=dict(phase=0) ,transform_param=dict(scale=1./255, mirror=True, mean_file = "examples/scene/scene_mean.binaryproto") ,ntop=2)

n.pair_data1, n.label1 = L.Data(name = 'pair_data',batch_size=testBatchSize, backend=P.Data.LMDB, source="examples/scene/test_pairs.lmdb", include=dict(phase=1) ,transform_param=dict(scale=1./255, mirror=True, mean_file = "examples/scene/scene_mean.binaryproto") ,ntop=2)

n.i1, n.i2 = L.Slice(n.pair_data,name = 'slice', slice_param=dict(slice_dim=1, slice_point=3),ntop=2)

n.p1 = L.Pooling(n.i1,name="pool_i1", kernel_size=2, stride=2, pool=P.Pooling.AVE)
n.p2 = L.Pooling(n.i2,name="pool_i2", kernel_size=2, stride=2, pool=P.Pooling.AVE)

n.crop_ref1=L.Input(input_param=dict(shape=dict(dim=[trainBatchSize,3,cropSize,cropSize])), include=dict(phase=0))
n.crop_ref2=L.Input(input_param=dict(shape=dict(dim=[testBatchSize,3,cropSize,cropSize])), include=dict(phase=1))

n.crop11=L.Crop(n.i1, n.crop_ref1, crop_param=dict(axis=2, offset=[0,0]))

n.conv1_11=L.Convolution(n.crop11, param=[dict(name="conv1_w", lr_mult=1), dict(name="conv1_b", lr_mult=2)], convolution_param=dict(num_output=20, kernel_size=5, stride=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))

n.pool1_11=maxPool(n.conv1_11)
	


#n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
#                     transform_param=dict(scale=1./255), ntop=2)

#n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
#n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
#n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
#n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
#n.relu1 = L.ReLU(n.fc1, in_place=True)
#n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
#n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    
    
with open('./gen.prototxt', 'w') as f:
    f.write(str(n.to_proto()))
    




