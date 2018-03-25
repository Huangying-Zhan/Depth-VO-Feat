#!/usr/bin/env sh

# Create a folder for snaptshots.
mkdir -p snapshots/depth_odometry

TOOLS=$YOUR_CAFFE_DIR/build/tools
$TOOLS/caffe train\
  --solver experiments/depth_odometry/solver.prototxt\
  --gpu 0\
  --weights snapshots/depth/train_iter_200000.caffemodel