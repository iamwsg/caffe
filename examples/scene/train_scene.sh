#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/scene/scene_solver_2.prototxt 2>&1 | tee scene_multi.log
#$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt

