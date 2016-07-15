#!/usr/bin/env sh

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt -weights examples/siamese/My_mnist_siamese_0to9l_iter_50000.caffemodel 2>&1 | tee ./examples/mnist/lenet_train_fine4.log

./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>&1 | tee ./examples/mnist/lenet_train_s.log
