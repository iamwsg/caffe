#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/scene/scene_solver_2.prototxt --weights=/home/shaogangwang/Downloads/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel  2>&1 | tee scene_multi.log

#$TOOLS/caffe train --solver=examples/scene/scene_solver_2.prototxt --weights=examples/scene/models/baseLine_train7_20000_pad_batch_400/scene_iter_10000.caffemodel 2>&1 | tee scene_multi.log



#$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt




#./convert_pairs.bin --resize_height=128 --resize_width=128 /home/shaogangwang/Datasets/ /home/shaogangwang/mywork/caffe/examples/scene/train11_pairs_300000_pad.txt /home/shaogangwang/mywork/caffe/examples/scene/train11_pairs_300000_pad.lmdb


#./convert_pairs.bin --resize_height=128 --resize_width=128 /home/shaogangwang/Datasets/ /home/shaogangwang/mywork/caffe/examples/scene/train7_pairs_40000_pad.txt /home/shaogangwang/mywork/caffe/examples/scene/train7_pairs_40000_pad.lmdb

#./convert_pairs.bin --resize_height=128 --resize_width=128 /home/shaogangwang/Datasets/ /home/shaogangwang/mywork/caffe/examples/scene/test_pairs_1000_pad.txt /home/shaogangwang/mywork/caffe/examples/scene/test_pairs_1000_pad.lmdb
