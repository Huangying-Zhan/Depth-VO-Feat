# Introduction

This repo implements the system described in the CVPR-2018 paper:

[**Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction** 
]() 

Huangying Zhan, Ravi Garg, Chamara Saroj Weerasekera, Kejie Li, Harsh Agarwal, Ian Reid

```
@article{zhan2018unsupervised,
  title={Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction},
  author={Zhan, Huangying and Garg, Ravi and Weerasekera, Chamara Saroj and Li, Kejie and Agarwal, Harsh and Reid, Ian},
  journal={arXiv preprint arXiv:1803.03893},
  year={2018}
}
```
This repo includes (1) the training procedure of our models;  (2) evaluation scripts for the results; (3) trained models and results.


### Contents
1. [Requirements](#part-1-requirements)
2. [Prepare dataset](#part-2-prepare-dataset)
3. [Depth](#part-3-depth)
4. [Depth and odometry](#part-4-depth-and-odometry)
5. [Feature Reconstruction Loss for Depth](#part-5-feature-reconstruction-loss-for-depth)
6. [Depth, odometry and feature](#part-6-depth-odometry-and-feature)
7. [Result evaluation](#part-7-result-evaluation)


### Part 1. Requirements

This code was tested with Python 2.7, CUDA 8.0 and Ubuntu 14.04 using [Caffe](http://caffe.berkeleyvision.org/).

Caffe: Add the required layers in `./caffe` into your own Caffe. Remember to enable Python Layers in the Caffe configuration.

Most of our required models, trained models and results can be downloaded from [here](https://www.dropbox.com/sh/qxfqflrrzzwupua/AAAPA1mF0QaKwwR2Ds0jtDhYa?dl=0). The following instruction also includes specific links to the items.

### Part 2. Download dataset and models

The main dataset used in this project is [KITTI Driving Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php). Please follow the instruction in `./data/README.md` to prepare the required dataset.

For our trained models and pre-requested models, please visit [here](https://www.dropbox.com/sh/60onn52jm9g2ygu/AADUkDRkwycS1STazstG5XOpa?dl=0) to download the models and put the models into the directory `./models`.

### Part 3. Depth

In this part, the training of single view depth estimation network from stereo pairs is introduced. Photometric loss is used as the main supervision signal. Only stereo pairs are used in this experiment.

1. Update `$YOUR_CAFFE_DIR` in `./experiments/depth/train.sh`. 
2. Run `bash ./expriments/depth/train.sh`. 

The trained models are saved in `./snapshots/depth`

### Part 4. Depth and odometry

In this part, the joint training of the depth estimation network and the visual odometry network is introduced. 
Photometric losses for spatial pairs and temporal pairs are used as the main supervision signal. 
Both spatial (stereo) pairs and temporal pairs (i.e. stereo sequences) are used in this experiment.

To facilitate the training, the model trained in the Depth experiment is used as an initialization.
1. Update `$YOUR_CAFFE_DIR` in `./experiments/depth_odometry/train.sh`. 
2. Run `bash ./expriments/depth_odometry/train.sh`. 

The trained models are saved in `./snapshots/depth_odometry`

### Part 5. Feature Reconstruction Loss for Depth 

In this part, the training of single view depth estimation network from stereo pairs is introduced. Both photometric loss and feature reconstruction loss are used as the main supervision signal. Only stereo pairs are used in this experiment. There are several features we have tried for this experiment. Currently, only the example of using **KITTI Feat.** is shown here. More details of using other features will be updated later.

To facilitate the training, the model trained in the Depth experiment is used as an initialization.
1. Update `$YOUR_CAFFE_DIR` in `./experiments/depth_feature/train.sh`. 
2. Run `bash ./expriments/depth_feature/train.sh`. 

The trained models are saved in `./snapshots/depth_feature`

### Part 6. Depth, odometry and feature

In this part, we show the training including feature reconstruction loss.
Stereo sequences are used in this experiment.

With the feature extractor proposed in [Weerasekera et.al](https://arxiv.org/abs/1711.05919), we can finetune the trained depth model and/or odometry model with our proposed deep feature reconstruction loss.

1. Update `$YOUR_CAFFE_DIR` in `./experiments/depth_odometry_feature/train.sh`. 
2. Run `bash ./expriments/depth_odometry_feature/train.sh`. 
 
**NOTE:** The link to download the feature extractor proposed in [Weerasekera et.al](https://arxiv.org/abs/1711.05919) will be released soon.

### Part 7. Result evalution

Note that the evaluation script provided here uses a different image interpolation for resizing input images (i.e. python's interpolation v.s. Caffe's interpolation), therefore the quantative result could be a little different from the published result. 

#### Depth estimation

Using the test set (697 image-depth pairs from 28 scenes) in Eigen Split is a common protocol to evaluate depth estimation result.

We basically use the evaluation script provided by [monodepth](https://github.com/mrharicot/monodepth) to evalute depth estimation results.

In order to run the evaluation, a `npy` file is required to store the predicted depths. Then run the script to evaluate the performance.

1. Update `caffe_root` in `./tools/evaluation_tools.py`
2. To generate the depth prediction and save it in a `npy` file. 
```
 python ./tools/evaluation_tools.py --func generate_depth_npy --dataset kitti_eigen --depth_net_def ./experiments/networks/depth_deploy.prototxt --model models/trained_models/Baseline.caffemodel --npy_dir ./result/depth/inv_depths_baseline.npy
```

3. To evalute the predictions.
```
python ./tools/eval_depth.py --split eigen --predicted_inv_depth_path ./result/depth/inv_depths_baseline.npy --gt_path data/kitti_raw_data/ --min_depth 1  --max_depth 50 --garg_crop
```

Some of our results (inverse depths) are released and can be downloaded from [here](https://www.dropbox.com/sh/1f6nkd4ezx0qfw4/AADmGuFLIxImtikz2UJrHeTOa?dl=0).

#### Visual Odometry

[KITTI Odometry benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) contains 22 stereo sequences, in which 11 sequences are provided with ground truth. The 11 sequences are used for evaluation or training of visual odometry. 

1. Update `caffe_root` in `./tools/evaluation_tools.py`
2. To generate the odometry predictions (relative camera motions), run the following script.

```
python ./tools/evaluation_tools.py --func generate_odom_result --model models/trained_models/Temporal.caffemodel --odom_net_def ./experiments/networks/odometry_deploy.prototxt --odom_result_dir ./result/odom_result
```

3. After getting the odometry predictions, we can evalute the performance by comparing with the ground truth poses.

```
python ./tools/evaluation_tools.py --func eval_odom --odom_result_dir ./result/odometry
```

### To-do list

- Dataset generator for KITTI Odometry split
- Details related to other features for feature reconstruction loss