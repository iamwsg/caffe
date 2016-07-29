caffe_root = '/home/shaogang/caffe/'  
folder=caffe_root+'/examples/scene/'
import sys
import os
sys.path.insert(0, caffe_root + 'python')
import caffe

from caffe import layers as L, params as P

def maxPool(bottom):
	top=L.Pooling(bottom, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	return top

def conv(bottom, numOutput, nameW, nameB):
	top=L.Convolution(bottom, param=[dict(name=nameW, lr_mult=1), dict(name=nameB, lr_mult=2)], 		convolution_param=dict(num_output=numOutput, kernel_size=5, stride=1, weight_filler=dict(type='xavier'), 		bias_filler=dict(type='constant')))
	return top

def ip(bottom, numOutput, nameW, nameB):
	top=L.InnerProduct(bottom, param=[dict(name=nameW, lr_mult=1), dict(name=nameB, lr_mult=2)], 			inner_product_param=dict(num_output=numOutput, weight_filler=dict(type='xavier'), 
		bias_filler=dict(type='constant')))
	return top

def reLu(bottom):
	top=L.ReLU(bottom, in_place=True)
	return top

def tower(bottom):
		conv1=conv(bottom,20,"conv1_w","conv1_b")
		pool1=maxPool(conv1)
		conv2=conv(pool1,50,"conv2_w","conv2_b")
		pool2=maxPool(conv2)
		ip1=ip(pool2,500,"ip1_w","ip1_b")
		reLu1=reLu(ip1)
		ip2=ip(reLu1,500,"ip2_w","ip2_b")
		return ip2

def towerMini(bottom):
		conv1=conv(bottom,20,"conv1_w","conv1_b")
		pool1=maxPool(conv1)
		conv2=conv(pool1,50,"conv2_w","conv2_b")
		pool2=maxPool(conv2)
		ip1=ip(pool2,100,"ip1_w","ip1_b")
		reLu1=reLu(ip1)
		ip2=ip(reLu1,50,"ip2_w","ip2_b")
		return ip2


def concat(bottom1,bottom2):
	return L.Concat(bottom1,bottom2, concat_param=dict(axis=1))

def doubleTower(bottom1,bottom2):
		t1=tower(bottom1)
		t2=tower(bottom2)
		con=concat(t1,t2)
		ip1=ip(con,512,"fc1_w","fc1_b")
		relu1=reLu(ip1)
		ip2=ip(relu1,512,"fc2_w","fc2_b")
		relu2=reLu(ip2)
		ip3=ip(relu2,1,"fc3_w","fc3_b")
		return ip3

def doubleTowerMini(bottom1,bottom2):
		t1=towerMini(bottom1)
		t2=towerMini(bottom2)
		con=concat(t1,t2)
		ip1=ip(con,64,"fc1_w","fc1_b")
		relu1=reLu(ip1)
		ip2=ip(relu1,32,"fc2_w","fc2_b")
		relu2=reLu(ip2)
		ip3=ip(relu2,1,"fc3_w","fc3_b")
		return ip3


def concatN(b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19):
	return L.Concat(b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19, concat_param=dict(axis=-1))

def concat3(b1,b2,b3):
	return L.Concat(b1,b2,b3,concat_param=dict(axis=-1))

def padLabel(b1,b2):
	return L.Concat(b1,b2,concat_param=dict(axis=1))

def reshape(bottom,Dim):
	return L.Reshape(bottom, reshape_param=dict(shape=dict(dim=Dim)))

def unevenPool(bottom,hight,width,poolType):
	top=L.Pooling(bottom, kernel_h=hight,kernel_w=width, stride=1, pool=poolType)
	return top

def softMaxLoss(netOutput,label):
	return L.SoftmaxWithLoss(netOutput, label)

def hingeLoss(netOutput,label):
	return L.HingeLoss(netOutput, label)

def acc(netOutput, label, Phase):
	return L.Accuracy(netOutput, label, include=dict(phase=Phase))

def data(src, mean, numBatch, Phase):
	pair_data, label = L.Data(batch_size=numBatch, backend=P.Data.LMDB, source=src, include=dict(phase=Phase) ,transform_param=dict(scale=1./255, mirror=True, mean_file = mean) ,ntop=2)
	return pair_data, label
	
def sliceData(bottom):
	i1,i2=L.Slice(bottom, slice_param=dict(slice_dim=1, slice_point=3),ntop=2)
	return i1,i2

def avePool(bottom):
	return L.Pooling(bottom, kernel_size=2, stride=2, pool=P.Pooling.AVE)

def crop(bottom,Phase,numBatch,Axis,Offset,cropSize):
	crop_ref=L.Input(input_param=dict(shape=dict(dim=[numBatch,3,cropSize,cropSize])), 
				include=dict(phase=Phase))
	top=L.Crop(bottom, crop_ref, crop_param=dict(axis=Axis, offset=Offset))
	return top


def matchNetTrain(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.c11=crop(trNet.i1,Phase,trainBatchSize,2,[0,0],cropSize)
	trNet.c12=crop(trNet.i1,Phase,trainBatchSize,2,[0,32],cropSize)
	trNet.c13=crop(trNet.i1,Phase,trainBatchSize,2,[0,64],cropSize)
	trNet.c14=crop(trNet.i1,Phase,trainBatchSize,2,[32,0],cropSize)
	trNet.c15=crop(trNet.i1,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.c16=crop(trNet.i1,Phase,trainBatchSize,2,[32,64],cropSize)
	trNet.c17=crop(trNet.i1,Phase,trainBatchSize,2,[64,0],cropSize)
	trNet.c18=crop(trNet.i1,Phase,trainBatchSize,2,[64,32],cropSize)
	trNet.c19=crop(trNet.i1,Phase,trainBatchSize,2,[64,64],cropSize)
	trNet.c21=crop(trNet.i2,Phase,trainBatchSize,2,[0,0],cropSize)
	trNet.c22=crop(trNet.i2,Phase,trainBatchSize,2,[0,32],cropSize)
	trNet.c23=crop(trNet.i2,Phase,trainBatchSize,2,[0,64],cropSize)
	trNet.c24=crop(trNet.i2,Phase,trainBatchSize,2,[32,0],cropSize)
	trNet.c25=crop(trNet.i2,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.c26=crop(trNet.i2,Phase,trainBatchSize,2,[32,64],cropSize)
	trNet.c27=crop(trNet.i2,Phase,trainBatchSize,2,[64,0],cropSize)
	trNet.c28=crop(trNet.i2,Phase,trainBatchSize,2,[64,32],cropSize)
	trNet.c29=crop(trNet.i2,Phase,trainBatchSize,2,[64,64],cropSize)
	trNet.dt0=doubleTower(trNet.p1,trNet.p2)
	trNet.dt1=doubleTower(trNet.p1,trNet.c21)
	trNet.dt2=doubleTower(trNet.p1,trNet.c22)
	trNet.dt3=doubleTower(trNet.p1,trNet.c23)
	trNet.dt4=doubleTower(trNet.p1,trNet.c24)
	trNet.dt5=doubleTower(trNet.p1,trNet.c25)
	trNet.dt6=doubleTower(trNet.p1,trNet.c26)
	trNet.dt7=doubleTower(trNet.p1,trNet.c27)
	trNet.dt8=doubleTower(trNet.p1,trNet.c28)
	trNet.dt9=doubleTower(trNet.p1,trNet.c29)
	trNet.dt10=doubleTower(trNet.p2,trNet.c11)
	trNet.dt11=doubleTower(trNet.p2,trNet.c12)
	trNet.dt12=doubleTower(trNet.p2,trNet.c13)
	trNet.dt13=doubleTower(trNet.p2,trNet.c14)
	trNet.dt14=doubleTower(trNet.p2,trNet.c15)
	trNet.dt15=doubleTower(trNet.p2,trNet.c16)
	trNet.dt16=doubleTower(trNet.p2,trNet.c17)
	trNet.dt17=doubleTower(trNet.p2,trNet.c18)
	trNet.dt18=doubleTower(trNet.p2,trNet.c19)
	trNet.con=concatN(trNet.dt0,trNet.dt1,trNet.dt2,trNet.dt3,trNet.dt4,trNet.dt5,trNet.dt6,trNet.dt7,
			trNet.dt8,trNet.dt9,trNet.dt10,trNet.dt11,trNet.dt12,trNet.dt13,trNet.dt14,trNet.dt15,
			trNet.dt16,trNet.dt17,trNet.dt18)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,19, P.Pooling.AVE)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.loss=hingeLoss(trNet.r2,trNet.label)
	trNet.accuracy=acc(trNet.r2, trNet.label, Phase)
	return trNet

def matchNetSimple(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.dt=doubleTower(trNet.p1,trNet.p2)
	trNet.accuracy=acc(trNet.dt, trNet.label, Phase)
	trNet.loss=hingeLoss(trNet.dt,trNet.label)
	return trNet

def matchNetMini(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.c11=crop(trNet.i1,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.c21=crop(trNet.i2,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.dt=doubleTowerMini(trNet.p1, trNet.p2)
	trNet.dt1=doubleTowerMini(trNet.p1, trNet.c21)
	trNet.dt2=doubleTowerMini(trNet.p2, trNet.c11)
	trNet.con=concat3(trNet.dt,trNet.dt1,trNet.dt2)
	trNet.r1=reshape(trNet.con,[0,1,2,-1])
	trNet.p=unevenPool(trNet.r1,1,3, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,2,1,-1])
	trNet.accuracy=acc(trNet.r2, trNet.label, Phase)
	trNet.loss=hingeLoss(trNet.r2,trNet.label)
	return trNet

def matchNetMiniHinge(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.c11=crop(trNet.i1,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.c21=crop(trNet.i2,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.dt=doubleTowerMini(trNet.p1, trNet.p2)
	trNet.dt1=doubleTowerMini(trNet.p1, trNet.c21)
	trNet.dt2=doubleTowerMini(trNet.p2, trNet.c11)
	trNet.con=concat3(trNet.dt,trNet.dt1,trNet.dt2)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,3, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1])
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.accuracy=acc(trNet.pad, trNet.label, Phase)
	trNet.loss=hingeLoss(trNet.pad,trNet.label)
	return trNet



#trainSrc="examples/scene/train_pairs.lmdb"
#testSrc="examples/scene/test_pairs.lmdb"

trainSrc="examples/scene/train_pairs_unseen_hinge.lmdb"
testSrc="examples/scene/test_pairs_unseen_hinge.lmdb"
mean="examples/scene/scene_mean.binaryproto"

trainBatchSize=128
testBatchSize=100
cropSize=64

#trNet=matchNetTrain(trainSrc, mean, trainBatchSize, cropSize,0)
#teNet=matchNetTrain(testSrc, mean, testBatchSize, cropSize,1)

#trNetSimple=matchNetSimple(trainSrc, mean, trainBatchSize, cropSize,0)
#teNetSimple=matchNetSimple(testSrc, mean, testBatchSize, cropSize,1)

trNetMini=matchNetMiniHinge(trainSrc, mean, trainBatchSize/4, cropSize,0)
teNetMini=matchNetMiniHinge(testSrc, mean, testBatchSize/2, cropSize,1)

#with open('./matchNetTrainHinge.prototxt', 'w') as f:
#    f.write(str(trNet.to_proto()))

#with open('./matchNetTestHinge.prototxt', 'w') as f:
#    f.write(str(teNet.to_proto()))

#with open('./matchNetTrainHingeSimple.prototxt', 'w') as f:
#    f.write(str(trNetSimple.to_proto()))

#with open('./matchNetTestHingeMini.prototxt', 'w') as f:
#    f.write(str(teNetSimple.to_proto()))

with open('./matchNetTrainHingeMini.prototxt', 'w') as f:
    f.write(str(trNetMini.to_proto()))

with open('./matchNetTestHingeMini.prototxt', 'w') as f:
    f.write(str(teNetMini.to_proto()))



    
    
    

    




