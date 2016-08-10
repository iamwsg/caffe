#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/scene/scene_solver_met.prototxt 2>&1 | tee met.log
#$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt




#./convert_pairs.bin --resize_height=128 --resize_width=128 /home/shaogangwang/Datasets/ /home/shaogangwang/mywork/caffe/examples/scene/train11_pairs_300000_pad.txt /home/shaogangwang/mywork/caffe/examples/scene/train11_pairs_300000_pad.lmdb

