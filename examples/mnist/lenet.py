from pylab import *
import matplotlib

caffe_root = '../..'  

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import os
os.chdir(caffe_root)

#from caffe import layers as L, params as P


caffe.set_mode_cpu()

solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)

#solver = caffe.SGDSolver('examples/siamese/mnist_siamese_solver.prototxt')
solver = caffe.SGDSolver('examples/mnist/lenet_solver.prototxt')

#[(k, v.data.shape) for k, v in solver.net.blobs.items()]

solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

solver.step(100)
loss=solver.net.blobs['loss'].data;loss
acc=solver.test_nets[0].blobs['accuracy'].data;acc
ip2=solver.test_nets[0].blobs['ip2'].data;ip2
label=solver.test_nets[0].blobs['label'].data;label

#imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
#print 'train labels:', solver.net.blobs['label'].data[:8]

