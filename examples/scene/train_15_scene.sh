#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/15_scene/15_scene_solver.prototxt -weights models/bvlc_alexnet/bvlc_reference_caffenet.caffemodel -gpu 0
