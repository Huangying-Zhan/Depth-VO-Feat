#!/usr/bin/env sh

# Create a folder for snaptshots.
mkdir -p snapshots/depth_feature

TOOLS=$YOUR_CAFFE_DIR/build/tools
$TOOLS/caffe train\
  --solver experiments/depth_feature/solver.prototxt\
  --gpu 0\
  --weights models/feature_extractor/feat_extractor_KITTI_Feat.caffemodel,snapshots/depth/train_iter_200000.caffemodel 