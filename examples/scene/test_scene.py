from pylab import *
#import matplotlib
#import numpy as np
#########################################
#caffe_root = '/home/shaogang/caffe/'  
caffe_root = '/home/shaogangwang/mywork/caffe/'
#########################################  
folder=caffe_root+'examples/scene/'
import sys
import os
sys.path.insert(0, caffe_root + 'python')
import caffe

os.chdir(caffe_root)

#############################
caffe.set_mode_gpu()
#############################

solver = caffe.SGDSolver(folder+'scene_solver_2.prototxt')

solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)
solver.step(1)

label=solver.net.blobs['label'].data
th=solver.net.blobs['th'].data
pad=solver.net.blobs['pad'].data
loss=solver.net.blobs['loss'].data
label[:10],pad[:10,:,0,0]

dt=solver.net.blobs['dt'].data
con=solver.net.blobs['con'].data
r1=solver.net.blobs['r1'].data
r2=solver.net.blobs['r2'].data
label=solver.net.blobs['label'].data
r1[:5,0,:,:];r2[:5,:,0,0];label[:5]




rt1=solver.test_nets[0].blobs['r1'].data
rt2=solver.test_nets[0].blobs['r2'].data
labelt=solver.test_nets[0].blobs['label'].data
rt1[:5,0,:,:];rt2[:5,:,0,0];labelt[:5]

acc=solver.test_nets[0].blobs['accuracy'].data;acc
#out=solver.test_nets[0].blobs['out'].data;out



