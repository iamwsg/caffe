#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/15_scene
DATA=data/15_scene
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/15_scene_train_lmdb \
  $DATA/15_scene_mean.binaryproto

echo "Done."
