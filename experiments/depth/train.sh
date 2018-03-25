#!/usr/bin/env sh

# Create a folder for snaptshots.
mkdir -p snapshots/depth

# TOOLS=$YOUR_CAFFE_DIR/build/tools
TOOLS=/home/hyzhan/caffe/build/tools
$TOOLS/caffe train\
  --solver ./experiments/depth/solver.prototxt\
  --weights models/resnet_50_1by2.caffemodel
  --gpu 0