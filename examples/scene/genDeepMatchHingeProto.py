caffe_root = '/home/shaogang/caffe/'  
folder=caffe_root+'/examples/scene/'
import sys
import os
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

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

def towerMiniConv3(bottom):
		conv1=conv(bottom,50,"conv1_w","conv1_b") #20
		pool1=maxPool(conv1)
		conv2=conv(pool1,100,"conv2_w","conv2_b") #50
		pool2=maxPool(conv2)
		conv3=conv(pool2,100,"conv3_w","conv3_b") #50
		pool3=maxPool(conv3)
		ip1=ip(pool3,200,"ip1_w","ip1_b") #100
		reLu1=reLu(ip1)
		ip2=ip(reLu1,100,"ip2_w","ip2_b") #50
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
		ip3=ip(relu2,2,"fc3_w","fc3_b")
		return ip3

def doubleTowerMini_1(bottom1,bottom2):
		t1=towerMini(bottom1)
		t2=towerMini(bottom2)
		con=concat(t1,t2)
		ip1=ip(con,64,"fc1_w","fc1_b")
		relu1=reLu(ip1)
		ip2=ip(relu1,32,"fc2_w","fc2_b")
		relu2=reLu(ip2)
		ip3=ip(relu2,1,"fc3_w","fc3_b")
		return ip3

def doubleTowerMini_1_conv3(bottom1,bottom2):
		t1=towerMiniConv3(bottom1)
		t2=towerMiniConv3(bottom2)
		con=concat(t1,t2)
		ip1=ip(con,200,"fc1_w","fc1_b") #64
		relu1=reLu(ip1)
		#drop1=drop(relu1,"drop1",0.5)
		ip2=ip(relu1,100,"fc2_w","fc2_b") #32
		relu2=reLu(ip2)
		#drop2=drop(relu2,"drop2",0.5)
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

def data2(src, numBatch, Phase):
	pair_data, label = L.Data(batch_size=numBatch, backend=P.Data.LMDB, source=src, include=dict(phase=Phase) ,transform_param=dict(scale=1, mirror=False) ,ntop=2)
	return pair_data, label
	
def sliceData(bottom):
	i1,i2=L.Slice(bottom, slice_param=dict(slice_dim=1, slice_point=3),ntop=2)
	return i1,i2

def sliceData3(bottom):
	i1,i2,i3=L.Slice(bottom, slice_param=dict(slice_dim=1, slice_point=[3,4]),ntop=3)
	return i1,i2,i3

def sliceData20(bottom):
	i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20=L.Slice(bottom, slice_param=dict(slice_dim=2, slice_point=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]),ntop=20)
	return i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20


def sliceDataN(bottom,n):
	i=L.Slice(bottom, slice_param=dict(slice_dim=1, slice_point=np.arange(n)+1),ntop=n)
	return i


#def sliceData20(bottom):
#	i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20=L.Slice(bottom, slice_param=dict(slice_dim=1, slice_point=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]),ntop=20)
#	return i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20

def avePool(bottom):
	return L.Pooling(bottom, kernel_size=2, stride=2, pool=P.Pooling.AVE)

def crop(bottom,Phase,numBatch,Axis,Offset,cropSize):
	crop_ref=L.Input(input_param=dict(shape=dict(dim=[numBatch,3,cropSize,cropSize])), 
				include=dict(phase=Phase))
	top=L.Crop(bottom, crop_ref, crop_param=dict(axis=Axis, offset=Offset))
	return top

def threshold(bottom, th):
	return L.Threshold(bottom, threshold_param=dict(threshold=th))


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
	trNet.dt0=doubleTowerMini(trNet.p1,trNet.p2)
	trNet.dt1=doubleTowerMini(trNet.p1,trNet.c21)
	trNet.dt2=doubleTowerMini(trNet.p1,trNet.c22)
	trNet.dt3=doubleTowerMini(trNet.p1,trNet.c23)
	trNet.dt4=doubleTowerMini(trNet.p1,trNet.c24)
	trNet.dt5=doubleTowerMini(trNet.p1,trNet.c25)
	trNet.dt6=doubleTowerMini(trNet.p1,trNet.c26)
	trNet.dt7=doubleTowerMini(trNet.p1,trNet.c27)
	trNet.dt8=doubleTowerMini(trNet.p1,trNet.c28)
	trNet.dt9=doubleTowerMini(trNet.p1,trNet.c29)
	trNet.dt10=doubleTowerMini(trNet.p2,trNet.c11)
	trNet.dt11=doubleTowerMini(trNet.p2,trNet.c12)
	trNet.dt12=doubleTowerMini(trNet.p2,trNet.c13)
	trNet.dt13=doubleTowerMini(trNet.p2,trNet.c14)
	trNet.dt14=doubleTowerMini(trNet.p2,trNet.c15)
	trNet.dt15=doubleTowerMini(trNet.p2,trNet.c16)
	trNet.dt16=doubleTowerMini(trNet.p2,trNet.c17)
	trNet.dt17=doubleTowerMini(trNet.p2,trNet.c18)
	trNet.dt18=doubleTowerMini(trNet.p2,trNet.c19)
	trNet.con=concatN(trNet.dt0,trNet.dt1,trNet.dt2,trNet.dt3,trNet.dt4,trNet.dt5,trNet.dt6,trNet.dt7,
			trNet.dt8,trNet.dt9,trNet.dt10,trNet.dt11,trNet.dt12,trNet.dt13,trNet.dt14,trNet.dt15,
			trNet.dt16,trNet.dt17,trNet.dt18)
	trNet.r1=reshape(trNet.con,[0,1,2,-1])
	trNet.p=unevenPool(trNet.r1,1,19, P.Pooling.AVE)	
	trNet.r2=reshape(trNet.p,[0,2,1,-1])
	trNet.loss=hingeLoss(trNet.r2,trNet.label)
	trNet.accuracy=acc(trNet.r2, trNet.label, Phase)
	return trNet

##19 pipelines with paded labels
def matchNetTrainPad(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
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
	trNet.dt0=doubleTowerMini_1(trNet.p1,trNet.p2)
	trNet.dt1=doubleTowerMini_1(trNet.p1,trNet.c21)
	trNet.dt2=doubleTowerMini_1(trNet.p1,trNet.c22)
	trNet.dt3=doubleTowerMini_1(trNet.p1,trNet.c23)
	trNet.dt4=doubleTowerMini_1(trNet.p1,trNet.c24)
	trNet.dt5=doubleTowerMini_1(trNet.p1,trNet.c25)
	trNet.dt6=doubleTowerMini_1(trNet.p1,trNet.c26)
	trNet.dt7=doubleTowerMini_1(trNet.p1,trNet.c27)
	trNet.dt8=doubleTowerMini_1(trNet.p1,trNet.c28)
	trNet.dt9=doubleTowerMini_1(trNet.p1,trNet.c29)
	trNet.dt10=doubleTowerMini_1(trNet.p2,trNet.c11)
	trNet.dt11=doubleTowerMini_1(trNet.p2,trNet.c12)
	trNet.dt12=doubleTowerMini_1(trNet.p2,trNet.c13)
	trNet.dt13=doubleTowerMini_1(trNet.p2,trNet.c14)
	trNet.dt14=doubleTowerMini_1(trNet.p2,trNet.c15)
	trNet.dt15=doubleTowerMini_1(trNet.p2,trNet.c16)
	trNet.dt16=doubleTowerMini_1(trNet.p2,trNet.c17)
	trNet.dt17=doubleTowerMini_1(trNet.p2,trNet.c18)
	trNet.dt18=doubleTowerMini_1(trNet.p2,trNet.c19)
	trNet.con=concatN(trNet.dt0,trNet.dt1,trNet.dt2,trNet.dt3,trNet.dt4,trNet.dt5,trNet.dt6,trNet.dt7,
			trNet.dt8,trNet.dt9,trNet.dt10,trNet.dt11,trNet.dt12,trNet.dt13,trNet.dt14,trNet.dt15,
			trNet.dt16,trNet.dt17,trNet.dt18)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,19, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1])
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	return trNet

##19 pipelines with paded labels with 3 conv layers
def matchNetTrainPad_conv3(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
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
	trNet.dt0=doubleTowerMini_1_conv3(trNet.p1,trNet.p2)
	trNet.dt1=doubleTowerMini_1_conv3(trNet.p1,trNet.c21)
	trNet.dt2=doubleTowerMini_1_conv3(trNet.p1,trNet.c22)
	trNet.dt3=doubleTowerMini_1_conv3(trNet.p1,trNet.c23)
	trNet.dt4=doubleTowerMini_1_conv3(trNet.p1,trNet.c24)
	trNet.dt5=doubleTowerMini_1_conv3(trNet.p1,trNet.c25)
	trNet.dt6=doubleTowerMini_1_conv3(trNet.p1,trNet.c26)
	trNet.dt7=doubleTowerMini_1_conv3(trNet.p1,trNet.c27)
	trNet.dt8=doubleTowerMini_1_conv3(trNet.p1,trNet.c28)
	trNet.dt9=doubleTowerMini_1_conv3(trNet.p1,trNet.c29)
	trNet.dt10=doubleTowerMini_1_conv3(trNet.p2,trNet.c11)
	trNet.dt11=doubleTowerMini_1_conv3(trNet.p2,trNet.c12)
	trNet.dt12=doubleTowerMini_1_conv3(trNet.p2,trNet.c13)
	trNet.dt13=doubleTowerMini_1_conv3(trNet.p2,trNet.c14)
	trNet.dt14=doubleTowerMini_1_conv3(trNet.p2,trNet.c15)
	trNet.dt15=doubleTowerMini_1_conv3(trNet.p2,trNet.c16)
	trNet.dt16=doubleTowerMini_1_conv3(trNet.p2,trNet.c17)
	trNet.dt17=doubleTowerMini_1_conv3(trNet.p2,trNet.c18)
	trNet.dt18=doubleTowerMini_1_conv3(trNet.p2,trNet.c19)
	trNet.con=concatN(trNet.dt0,trNet.dt1,trNet.dt2,trNet.dt3,trNet.dt4,trNet.dt5,trNet.dt6,trNet.dt7,
			trNet.dt8,trNet.dt9,trNet.dt10,trNet.dt11,trNet.dt12,trNet.dt13,trNet.dt14,trNet.dt15,
			trNet.dt16,trNet.dt17,trNet.dt18)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,19, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1])
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	return trNet

#three pipe lines
def matchNetMiniHingePadTrain(trainSrc,mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.c11=crop(trNet.i1,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.c21=crop(trNet.i2,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.dt=doubleTowerMini_1(trNet.p1, trNet.p2)
	trNet.dt1=doubleTowerMini_1(trNet.p1, trNet.c21)
	trNet.dt2=doubleTowerMini_1(trNet.p2, trNet.c11)
	trNet.con=concat3(trNet.dt,trNet.dt1,trNet.dt2)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,3, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1]) #
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
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
	trNet.p=unevenPool(trNet.r1,1,3, P.Pooling.AVE)	
	trNet.r2=reshape(trNet.p,[0,2,1,-1])
	trNet.accuracy=acc(trNet.r2, trNet.label, Phase)
	trNet.loss=hingeLoss(trNet.r2,trNet.label)
	return trNet



def matchNetMiniHingePadTest(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.c11=crop(trNet.i1,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.c21=crop(trNet.i2,Phase,trainBatchSize,2,[32,32],cropSize)
	trNet.dt=doubleTowerMini_1(trNet.p1, trNet.p2)
	trNet.dt1=doubleTowerMini_1(trNet.p1, trNet.c21)
	trNet.dt2=doubleTowerMini_1(trNet.p2, trNet.c11)
	trNet.con=concat3(trNet.dt,trNet.dt1,trNet.dt2)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,3, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1]) 
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.accuracy=acc(trNet.pad, trNet.label, Phase)
	trNet.loss=hingeLoss(trNet.pad,trNet.label)
	return trNet




def matchNetBaseLine(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.dt=doubleTowerMini(trNet.p1,trNet.p2)
	trNet.accuracy=acc(trNet.dt, trNet.label, Phase)
	trNet.loss=hingeLoss(trNet.dt,trNet.label)
	return trNet

def matchNetBaseLinePad(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.dt=doubleTowerMini(trNet.p1,trNet.p2)
	trNet.accuracy=acc(trNet.dt, trNet.th, Phase)
	trNet.loss=hingeLoss(trNet.dt,trNet.th)
	return trNet

def matchNetBaseLinePadoneSideHinge(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.dt=doubleTowerMini_1(trNet.p1,trNet.p2)
	trNet.r2=reshape(trNet.dt,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1]) 
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
	return trNet

def matchNetBaseLinePadConv3(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
	trNet.i1, trNet.i2=sliceData(trNet.data)
	trNet.p1=avePool(trNet.i1)
	trNet.p2=avePool(trNet.i2)
	trNet.dt=doubleTowerMini_1_conv3(trNet.p1,trNet.p2)
	trNet.r2=reshape(trNet.dt,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1]) 
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
	return trNet



def aconv(bottom, na ,numOutput, l1,d1,l2,d2, Pad, Group, kSize, st, Std, Val):
	top=L.Convolution(bottom, name=na, param=[dict(lr_mult=l1, decay_mult=d1), dict(lr_mult=l2, decay_mult=d2)], 		convolution_param=dict(num_output=numOutput, kernel_size=kSize, pad=Pad, group=Group, stride=st, weight_filler=dict(type="gaussian", std=Std), 		bias_filler=dict(type="constant", value=Val)))
	return top


def norm(bottom, na, lSize, al,be):
	return L.LRN(bottom, name=na, lrn_param=dict(local_size=lSize, alpha=al, beta=be))

def apool(bottom, na, pType, kSize, st):
	return L.Pooling(bottom, name=na, kernel_size=kSize, stride=st, pool=pType)


def aip(bottom, na, numOutput, l1,d1,l2,d2,Std,Const):
	top=L.InnerProduct(bottom, param=[dict(lr_mult=l1, decay_mult=d1), dict(lr_mult=l2, decay_mult=d2)], 			inner_product_param=dict(num_output=numOutput, weight_filler=dict(type="gaussian", std=Std), 
		bias_filler=dict(type="constant", value=Const)))
	return top

def drop(bottom, na, dropRatio):
	return L.Dropout(bottom, name=na, dropout_param=dict(dropout_ratio=0.5))


def alex():
	net=caffe.NetSpec()
	net.data, net.label = data(trainSrc,mean,trainBatchSize,0)
	net.conv1=aconv(net.data, "conv1", 96, 1,1,2,0,0,1,11,4,0.01,0)
	net.relu1=reLu(net.conv1)
	net.norm1=norm(net.relu1,"norm1",5,0.0001,0.75)
	net.pool1=apool(net.norm1, "pool1", P.Pooling.MAX, 3,2)
	net.conv2=aconv(net.pool1,"conv2",256,1,1,2,0,2,2,5,1,0.01,0.1)
	net.relu2=reLu(net.conv2)
	net.norm2=norm(net.relu2,"norm2",5,0.0001,0.75)
	net.pool2=apool(net.norm2, "pool2", P.Pooling.MAX, 3,2)
	net.conv3=aconv(net.pool2,"conv3",384,1,1,2,0,1,1,3,1,0.01,0)
	net.relu3=reLu(net.conv3)
	net.conv4=aconv(net.relu3,"conv4",384,1,1,2,0,1,2,3,1,0.01,0.1)
	net.relu4=reLu(net.conv4)
	net.conv5=aconv(net.relu4,"conv5",256,1,1,2,0,1,2,3,1,0.01,0.1)
	net.relu5=reLu(net.conv5)
	net.pool5=apool(net.conv5, "pool5", P.Pooling.MAX, 3,2)
	net.fc6=aip(net.pool5,"fc6",4096, 1,1,2,0,0.005,0.1)
	net.relu6=reLu(net.fc6)
	net.drop6=drop(net.relu6,"drop6",0.5)
	net.fc7=aip(net.drop6,"fc7",4096, 1,1,2,0,0.005,0.1)
	net.relu7=reLu(net.fc7)
	return net

def alexTower(bottom):
	conv1=aconv(bottom, "conv1", 96, 1,1,2,0,0,1,11,4,0.01,0)
	relu1=reLu(conv1)
	norm1=norm(relu1,"norm1",5,0.0001,0.75)
	pool1=apool(norm1, "pool1", P.Pooling.MAX, 3,2)
	conv2=aconv(pool1,"conv2",256,1,1,2,0,2,2,5,1,0.01,0.1)
	relu2=reLu(conv2)
	norm2=norm(relu2,"norm2",5,0.0001,0.75)
	pool2=apool(norm2, "pool2", P.Pooling.MAX, 3,2)
	conv3=aconv(pool2,"conv3",384,1,1,2,0,1,1,3,1,0.01,0)
	relu3=reLu(conv3)
	conv4=aconv(relu3,"conv4",384,1,1,2,0,1,2,3,1,0.01,0.1)
	relu4=reLu(conv4)
	conv5=aconv(relu4,"conv5",256,1,1,2,0,1,2,3,1,0.01,0.1)
	relu5=reLu(conv5)
	pool5=apool(conv5, "pool5", P.Pooling.MAX, 3,2)
	fc6=aip(pool5,"fc6",4096, 1,1,2,0,0.005,0.1)
	relu6=reLu(fc6)
	drop6=drop(relu6,"drop6",0.5)
	fc7=aip(drop6,"fc7",4096, 1,1,2,0,0.005,0.1)
	relu7=reLu(fc7)
	return relu7

def alexTowerSmall(bottom):
	conv1=aconv(bottom, "conv1", 96, 0,0,0,0,5,1,11,4,0.01,0)
	relu1=reLu(conv1)
	norm1=norm(relu1,"norm1",5,0.0001,0.75)
	pool1=apool(norm1, "pool1", P.Pooling.MAX, 3,2)
	conv2=aconv(pool1,"conv2",256,0,0,0,0,2,2,5,1,0.01,0.1)
	relu2=reLu(conv2)
	norm2=norm(relu2,"norm2",5,0.0001,0.75)
	pool2=apool(norm2, "pool2", P.Pooling.MAX, 3,2)
	conv3=aconv(pool2,"conv3",384,0,0,0,0,1,1,3,1,0.01,0)
	relu3=reLu(conv3)
	conv4=aconv(relu3,"conv4",384,0,0,0,0,1,2,3,1,0.01,0.1)
	relu4=reLu(conv4)
	conv5=aconv(relu4,"conv5",256,0,0,0,0,1,2,3,1,0.01,0.1)
	relu5=reLu(conv5)
	pool5=apool(conv5, "pool5", P.Pooling.MAX, 3,2)
	fc6=ip(pool5,256,"fc6_w","fc6_b")
	relu6=reLu(fc6)
	#drop6=drop(relu6,"drop6",0.5)
	fc7=ip(relu6,256,"fc7_w","fc7_b")
	relu7=reLu(fc7)
	return relu7

def alexDoubleTower(bottom1,bottom2):
	t1=alexTowerSmall(bottom1)
	t2=alexTowerSmall(bottom2)
	con=concat(t1,t2)
	ip1=ip(con,256,"fc8_w","fc8_b")
	relu1=reLu(ip1)
	#drop1=drop(relu1,"drop1",0.5)
	ip2=ip(relu1,128,"fc9_w","fc9_b")
	relu2=reLu(ip2)
	#drop2=drop(relu2,"drop2",0.5)
	ip3=ip(relu2,1,"fc10_w","fc10_b")
	return ip3

##19 pipelines alex siamese net
def matchNetTrainPadAlex(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
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
	trNet.dt0=alexDoubleTower(trNet.p1,trNet.p2)
	trNet.dt1=alexDoubleTower(trNet.p1,trNet.c21)
	trNet.dt2=alexDoubleTower(trNet.p1,trNet.c22)
	trNet.dt3=alexDoubleTower(trNet.p1,trNet.c23)
	trNet.dt4=alexDoubleTower(trNet.p1,trNet.c24)
	trNet.dt5=alexDoubleTower(trNet.p1,trNet.c25)
	trNet.dt6=alexDoubleTower(trNet.p1,trNet.c26)
	trNet.dt7=alexDoubleTower(trNet.p1,trNet.c27)
	trNet.dt8=alexDoubleTower(trNet.p1,trNet.c28)
	trNet.dt9=alexDoubleTower(trNet.p1,trNet.c29)
	trNet.dt10=alexDoubleTower(trNet.p2,trNet.c11)
	trNet.dt11=alexDoubleTower(trNet.p2,trNet.c12)
	trNet.dt12=alexDoubleTower(trNet.p2,trNet.c13)
	trNet.dt13=alexDoubleTower(trNet.p2,trNet.c14)
	trNet.dt14=alexDoubleTower(trNet.p2,trNet.c15)
	trNet.dt15=alexDoubleTower(trNet.p2,trNet.c16)
	trNet.dt16=alexDoubleTower(trNet.p2,trNet.c17)
	trNet.dt17=alexDoubleTower(trNet.p2,trNet.c18)
	trNet.dt18=alexDoubleTower(trNet.p2,trNet.c19)
	trNet.con=concatN(trNet.dt0,trNet.dt1,trNet.dt2,trNet.dt3,trNet.dt4,trNet.dt5,trNet.dt6,trNet.dt7,
			trNet.dt8,trNet.dt9,trNet.dt10,trNet.dt11,trNet.dt12,trNet.dt13,trNet.dt14,trNet.dt15,
			trNet.dt16,trNet.dt17,trNet.dt18)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,19, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1])
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	return trNet



def mTower(bottom):
	conv0=aconv(bottom, "conv0", 24, 1,1,2,0,3,1,7,1,0.01,0)
	relu1=reLu(conv0)
	pool0=apool(relu1, "pool0", P.Pooling.MAX, 3,2)
	conv1=aconv(pool0, "conv1", 64, 1,1,2,0,2,1,5,1,0.01,0)
	relu2=reLu(conv1)
	pool1=apool(relu2, "pool1", P.Pooling.MAX, 3,2)
	conv2=aconv(pool1, "conv2", 96, 1,1,2,0,1,1,3,1,0.01,0)
	relu3=reLu(conv2)
	conv3=aconv(relu3, "conv3", 96, 1,1,2,0,1,1,3,1,0.01,0)
	relu4=reLu(conv3)
	conv4=aconv(relu4, "conv4", 64, 1,1,2,0,1,1,3,1,0.01,0)
	relu5=reLu(conv4)
	pool4=apool(relu5, "pool4", P.Pooling.MAX, 3,2)
	return pool4

def mDoubleTower(bottom1,bottom2):
		t1=mTower(bottom1)
		t2=mTower(bottom2)
		con=concat(t1,t2)
		ip1=ip(con,128,"fc1_w","fc1_b")
		relu1=reLu(ip1)
		ip2=ip(relu1,128,"fc2_w","fc2_b")
		relu2=reLu(ip2)
		ip3=ip(relu2,1,"fc3_w","fc3_b")
		return ip3


##19 pipelines with paded labels
def AmatchNetTrainPad(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
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
	trNet.dt0=alexDoubleTower(trNet.p1,trNet.p2)
	trNet.dt1=alexDoubleTower(trNet.p1,trNet.c21)
	trNet.dt2=alexDoubleTower(trNet.p1,trNet.c22)
	trNet.dt3=alexDoubleTower(trNet.p1,trNet.c23)
	trNet.dt4=alexDoubleTower(trNet.p1,trNet.c24)
	trNet.dt5=alexDoubleTower(trNet.p1,trNet.c25)
	trNet.dt6=alexDoubleTower(trNet.p1,trNet.c26)
	trNet.dt7=alexDoubleTower(trNet.p1,trNet.c27)
	trNet.dt8=alexDoubleTower(trNet.p1,trNet.c28)
	trNet.dt9=alexDoubleTower(trNet.p1,trNet.c29)
	trNet.dt10=alexDoubleTower(trNet.p2,trNet.c11)
	trNet.dt11=alexDoubleTower(trNet.p2,trNet.c12)
	trNet.dt12=alexDoubleTower(trNet.p2,trNet.c13)
	trNet.dt13=alexDoubleTower(trNet.p2,trNet.c14)
	trNet.dt14=alexDoubleTower(trNet.p2,trNet.c15)
	trNet.dt15=alexDoubleTower(trNet.p2,trNet.c16)
	trNet.dt16=alexDoubleTower(trNet.p2,trNet.c17)
	trNet.dt17=alexDoubleTower(trNet.p2,trNet.c18)
	trNet.dt18=alexDoubleTower(trNet.p2,trNet.c19)
	trNet.con=concatN(trNet.dt0,trNet.dt1,trNet.dt2,trNet.dt3,trNet.dt4,trNet.dt5,trNet.dt6,trNet.dt7,
			trNet.dt8,trNet.dt9,trNet.dt10,trNet.dt11,trNet.dt12,trNet.dt13,trNet.dt14,trNet.dt15,
			trNet.dt16,trNet.dt17,trNet.dt18)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,19, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1])
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	return trNet

def MmatchNetTrainPad(trainSrc, mean, trainBatchSize, cropSize, Phase):
	trNet=caffe.NetSpec()
	trNet.data, trNet.label = data(trainSrc,mean,trainBatchSize,Phase)
	trNet.th= threshold(trNet.label,0)
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
	trNet.dt0=mDoubleTower(trNet.p1,trNet.p2)
	trNet.dt1=mDoubleTower(trNet.p1,trNet.c21)
	trNet.dt2=mDoubleTower(trNet.p1,trNet.c22)
	trNet.dt3=mDoubleTower(trNet.p1,trNet.c23)
	trNet.dt4=mDoubleTower(trNet.p1,trNet.c24)
	trNet.dt5=mDoubleTower(trNet.p1,trNet.c25)
	trNet.dt6=mDoubleTower(trNet.p1,trNet.c26)
	trNet.dt7=mDoubleTower(trNet.p1,trNet.c27)
	trNet.dt8=mDoubleTower(trNet.p1,trNet.c28)
	trNet.dt9=mDoubleTower(trNet.p1,trNet.c29)
	trNet.dt10=mDoubleTower(trNet.p2,trNet.c11)
	trNet.dt11=mDoubleTower(trNet.p2,trNet.c12)
	trNet.dt12=mDoubleTower(trNet.p2,trNet.c13)
	trNet.dt13=mDoubleTower(trNet.p2,trNet.c14)
	trNet.dt14=mDoubleTower(trNet.p2,trNet.c15)
	trNet.dt15=mDoubleTower(trNet.p2,trNet.c16)
	trNet.dt16=mDoubleTower(trNet.p2,trNet.c17)
	trNet.dt17=mDoubleTower(trNet.p2,trNet.c18)
	trNet.dt18=mDoubleTower(trNet.p2,trNet.c19)
	trNet.con=concatN(trNet.dt0,trNet.dt1,trNet.dt2,trNet.dt3,trNet.dt4,trNet.dt5,trNet.dt6,trNet.dt7,
			trNet.dt8,trNet.dt9,trNet.dt10,trNet.dt11,trNet.dt12,trNet.dt13,trNet.dt14,trNet.dt15,
			trNet.dt16,trNet.dt17,trNet.dt18)
	trNet.r1=reshape(trNet.con,[0,1,1,-1])
	trNet.p=unevenPool(trNet.r1,1,19, P.Pooling.MAX)	
	trNet.r2=reshape(trNet.p,[0,1,1,-1])
	trNet.padL=reshape(trNet.label,[0,1,1,-1])
	trNet.pad=padLabel(trNet.r2,trNet.padL)
	trNet.loss=hingeLoss(trNet.pad,trNet.th)
	trNet.accuracy=acc(trNet.pad, trNet.th, Phase)
	return trNet

def metNet(b1,b2):
	con=concat(b1,b2)
	ip1=ip(con,256,"fc1_w","fc1_b")
	relu1=reLu(ip1)
	drop1=drop(relu1,"drop1",0.5)
	ip2=ip(drop1,128,"fc2_w","fc2_b")
	relu2=reLu(ip2)
	drop2=drop(relu2,"drop2",0.5)
	ip3=ip(drop2,1,"fc3_w","fc3_b")
	return ip3

def metricNet(src, trainBatchSize, Phase):
	net=caffe.NetSpec()
	net.data, net.label = data2(src,trainBatchSize,Phase)
	net.th= threshold(net.label,0)
	net.i1, net.i2, net.i3, net.i4, net.i5, net.i6, net.i7, net.i8, net.i9, net.i10, net.i11, net.i12, net.i13, net.i14, net.i15, net.i16, net.i17, net.i18, net.i19, net.i20=sliceData20(net.data)

	net.m1=metNet(net.i1,net.i11)
	net.m2=metNet(net.i1,net.i12)
	net.m3=metNet(net.i1,net.i13)
	net.m4=metNet(net.i1,net.i14)
	net.m5=metNet(net.i1,net.i15)
	net.m6=metNet(net.i1,net.i16)
	net.m7=metNet(net.i1,net.i17)
	net.m8=metNet(net.i1,net.i18)
	net.m9=metNet(net.i1,net.i19)
	net.m10=metNet(net.i1,net.i20)
	net.m11=metNet(net.i2,net.i11)
	net.m12=metNet(net.i3,net.i11)
	net.m13=metNet(net.i4,net.i11)
	net.m14=metNet(net.i5,net.i11)
	net.m15=metNet(net.i6,net.i11)
	net.m16=metNet(net.i7,net.i11)
	net.m17=metNet(net.i8,net.i11)
	net.m18=metNet(net.i9,net.i11)
	net.m19=metNet(net.i10,net.i11)
	net.con=concatN(net.m1,net.m2,net.m3,net.m4,net.m5,net.m6,net.m7,net.m8,
		net.m9,net.m10,net.m11,net.m12,net.m13,net.m14,net.m15,net.m16,net.m17,net.m18,net.m19)
	net.r1=reshape(net.con,[0,1,1,-1])
	net.p=unevenPool(net.r1,1,19, P.Pooling.MAX)	
	net.r2=reshape(net.p,[0,1,1,-1])
	net.padL=reshape(net.label,[0,1,1,-1])
	net.pad=padLabel(net.r2,net.padL)
	net.loss=hingeLoss(net.pad,net.th)
	net.accuracy=acc(net.pad, net.th, Phase)
	return net

padSrc="examples/scene/train_pairs_1000_pad.lmdb"
#padSrc="examples/scene/train11_pairs_300000_pad.lmdb"
#padSrc="examples/scene/scene_train7_pairs_20000_pad.lmdb"
#padSrc="examples/scene/train7_pairs_40000_pad.lmdb"
#padSrc="examples/scene/scene_train3_pairs_4000_pad.lmdb"
testSrc="examples/scene/test_pairs_1000_pad.lmdb"
#testSrc="examples/scene/test_pairs_1000_pad.lmdb"

#metricSrc="examples/scene/db"
#metricSrcTest="examples/scene/db_test"
#metricSrc="/home/shaogangwang/Datasets/FeatsDB/featsTrain10k"
#metricSrcTest="/home/shaogangwang/Datasets/FeatsDB/featsTest1k"
metricSrc="/home/shaogang/Datasets/FeatsDB/featsTrain20k"
metricSrcTest="/home/shaogang/Datasets/FeatsDB/featsTest1k"


mean="examples/scene/scene_mean.binaryproto"

trainBatchSize=100
testBatchSize=100
cropSize=64

#trNet=matchNetTrain(trainSrc, mean, trainBatchSize, cropSize,0)
#teNet=matchNetTrain(testSrc, mean, testBatchSize, cropSize,1)

#trNetSimple=matchNetSimple(trainSrc, mean, trainBatchSize, cropSize,0)
#teNetSimple=matchNetSimple(testSrc, mean, testBatchSize, cropSize,1)

#trNetMini=matchNetBaseLinePadConv3(padSrc, mean, trainBatchSize, cropSize,0)
#teNetMini=matchNetBaseLinePadConv3(testSrc, mean, testBatchSize, cropSize,1)

#trNetMini=matchNetTrain(trainSrc, mean, trainBatchSize, cropSize,0)
#teNetMini=matchNetTrain(testSrc, mean, testBatchSize, cropSize,1)

#trNetMini=matchNetTrainPad_conv3(padSrc ,mean, trainBatchSize, cropSize,0)
#teNetMini=matchNetTrainPad_conv3(testSrc, mean, testBatchSize, cropSize,1)

trNetMini=matchNetTrainPadAlex(padSrc ,mean, trainBatchSize, cropSize,0)
teNetMini=matchNetTrainPadAlex(testSrc, mean, testBatchSize, cropSize,1)


#trMetNet=metricNet(metricSrc, trainBatchSize, 0)
#teMetNet=metricNet(metricSrcTest, trainBatchSize, 1)
#with open('./metNetTrain.prototxt', 'w') as f:
#    f.write(str(trMetNet.to_proto()))
#with open('./metNetTest.prototxt', 'w') as f:
#    f.write(str(teMetNet.to_proto()))

#trNetMini=MmatchNetTrainPad(padSrc ,mean, trainBatchSize, cropSize,0)
#teNetMini=MmatchNetTrainPad(testSrc, mean, testBatchSize, cropSize,1)

#net=alex()
#with open('./alex.prototxt', 'w') as f:
#    f.write(str(net.to_proto()))

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



    
    

    




