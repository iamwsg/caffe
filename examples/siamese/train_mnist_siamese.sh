#!/usr/bin/env sh

TOOLS=./build/tools

<<<<<<< HEAD
$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt 2>&1 | tee siamese_train_09.log
=======
$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt 2>&1 | tee siamese_train08.log
>>>>>>> 4a3ee5864d5ddedb936e421c90fe671ee39b9b99
