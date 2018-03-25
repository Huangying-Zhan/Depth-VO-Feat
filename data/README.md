## Download dataset

The main dataset used in this project is [KITTI Driving Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php). The videos in three categories (City, Residential, Road) are used in our experiments.

<!-- FIXME: provide a script to download the dataset and arrange in required structure -->

## Dataset structure

After getting the dataset, in order to use our provided `dataset_builder.py`, the dataset should be arranged in the following structure. Let's call `$TOP` as the home directory of this repo. `kitti_raw_data` should be placed in the directory `$TOP/data` or a softlink should be created in the directory.

``` 
|-- $TOP
	|-- data
		|-- kitti_raw_data
			|-- city
		        |-- 2011_09_26_drive_0001
		        	|-- image_02 
		        		|-- data # contain left images
		        	|-- image_03 
		        		|-- data # contain right images
		        	|-- calib
		        		|-- calib_cam_to_cam.txt # contain camera intrinsic and extrainsic parameters
		        	|-- velodyne_points # contain laser readings for depth evaluation
		        	|-- oxts
		        |-- 2011_09_26_drive_0002
		        |-- ...
		    |-- residential
		    	|-- 2011_09_26_drive_0019
		        |-- ...
		    |-- road
		    	|-- 2011_09_26_drive_0015
		        |-- ...
```

## Build Dataset

### Train Set

We provide `dataset_builder.py` which builds the training set from the raw KITTI data. To use the script, please see the following example for creating dataset using [Eigen Split](https://arxiv.org/abs/1406.2283).

``` Shell
cd $TOP 
python data/dataset_builder.py --builder kitti_eigen --train_frame_distance 1 --raw_data_dir ./data/kitti_raw_data --dataset_dir ./data/dataset/kitti_eigen --image_size [160,608]
```

Other optional arguments/functions please refer to the script. **NOTE** if you have built a dataset and want to replace the original dataset, remember to delete the files in the original directory (especially **LMDB** folders, new LMDB file will be appended to the original file if you forgot to do so.). 

### Evaluation Set (Depth estimation)

The Eigen Split is commonly used for single view depth estimation benchmarking. 697 image-depth pairs are used for evalution. The list of images is saved at `./data/depth_evaluation/kitti_eigen/test_files_eigen.txt`. We also provide the images, which can be downloaded [here](https://www.dropbox.com/sh/n4uvg4rhdi4fzuk/AABWfmvc_WECj6h9X87M2d5Oa?dl=0) `./data/depth_evaluation/kitti_eigen`.

### Evaluation Set (Visual odometry)

[KITTI Odometry benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) contains 22 stereo sequences, in which 11 sequences are provided with ground truth. The 11 sequences are used for evaluation or training of visual odometry. For the details, please refer to our paper.

The ground truth files can be downloaded [here](http://www.cvlibs.net/download.php?file=data_odometry_poses.zip). The files should be arranged in the following structure.


``` 
|-- $TOP
	|-- data
		|-- kitti_raw_data
		|-- depth_evaluation
		|-- odometry_evaluation
			|--poses
				|-- 00.txt 
				|-- 01.txt
				|-- ...
				|-- 10.txt
```

