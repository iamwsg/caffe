#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/scene/scene_solver.prototxt 2>&1 | tee scene.log
#$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt

