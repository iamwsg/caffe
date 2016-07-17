#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt 2>&1 | tee siamese_train_06_test_89_feat_2_sim.log
#$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt

