from pylab import *
import matplotlib
#import numpy as np

caffe_root = '/home/shaogang/caffe/'  
#caffe_root = '/home/shaogangwang/mywork/caffe/'  
folder=caffe_root+'examples/scene/'
import sys
import os
sys.path.insert(0, caffe_root + 'python')
import caffe

os.chdir(caffe_root)

#MODEL_FILE = caffe_root+'examples/scene/scene.prototxt'
#PRETRAINED_FILE = caffe_root+ 'examples/scene/scene_iter_2000.caffemodel' 
#net = caffe.Net(MODEL_FILE,PRETRAINED_FILE,caffe.TEST)
#net = caffe.Net(MODEL_FILE,caffe.TEST)
#net_train = caffe.Net(MODEL_FILE, caffe.TEST)

caffe.set_mode_cpu()

solver = caffe.SGDSolver(folder+'scene_solver_2.prototxt')

solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)
solver.step(1)

dt=solver.net.blobs['dt'].data
con=solver.net.blobs['con'].data
r1=solver.net.blobs['r1'].data
r2=solver.net.blobs['r2'].data
r1[:5,0,:,:];r2[:5,:,0,0]
label=solver.net.blobs['label'].data

#print 'train labels:', solver.net.blobs['label'].data[:8]
#imshow(solver.net.blobs['pair_data'].data[0][0]);show()
feat=solver.net.blobs['feat'].data
label=solver.test_nets[0].blobs['label'].data
plot(feat[:,0],feat[:,1]);show()

niter = 10
test_interval = 10
# losses will also be stored in the log
train_loss = zeros(niter)
acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 100))


sim_test=solver.test_nets[0].blobs['label'].data.flatten();prob_test=np.round(solver.test_nets[0].blobs['prob'].data.flatten());mi=sim_test-prob_test;mi
loss=solver.net.blobs['loss'].data;loss
acc=solver.test_nets[0].blobs['accuracy'].data;acc
out=solver.test_nets[0].blobs['out'].data;out


# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    output[it] = solver.test_nets[0].blobs['prob'].data.flatten()
    
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    #output[it] = solver.test_nets[0].blobs['prob'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        for test_it in range(100):
            solver.test_nets[0].forward()
	    #acc[it]=solver.test_nets[0].blobs['accuracy'].data
            
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(acc)), acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(acc[-1]))
show()



