#!/usr/bin/env sh

# Create a folder for snaptshots.
mkdir -p snapshots/depth_odometry_feature

TOOLS=$YOUR_CAFFE_DIR/build/tools
$TOOLS/caffe train\
  --solver experiments/depth_odometry_feature/solver.prototxt\
  --gpu 0\
  # --weights snapshots/depth_odometry/train_iter_200000.caffemodel,model/weerasekera_nyu.caffemodel