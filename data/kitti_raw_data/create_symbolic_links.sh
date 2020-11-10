#!/bin/bash
KITTI_RAW_DATA="."
for category in city residential road
do
	echo "Category: $category"
	if [ ! -d "$category" ]; then
		mkdir "$category"
	fi
	for sequence in $(cat "${category}.txt")
	do
		echo "    Sequence: ${sequence}"
		if [ ! -e "${category}/${sequence}" ]; then
        	        mkdir "${category}/${sequence}"
	        fi
		if [ ! -e "${category}/${sequence}/image_02" ]; then
                        mkdir "${category}/${sequence}/image_02"
                fi
                if [ ! -e "${category}/${sequence}/image_03" ]; then
                        mkdir "${category}/${sequence}/image_03"
                fi
                if [ ! -e "${category}/${sequence}/calib" ]; then
                        mkdir "${category}/${sequence}/calib"
                fi
		kitti_day=$(echo $sequence | cut -c1-10)
	 	if [ ! -e "${category}/${sequence}/image_02/data" ]; then
			ln -s "${KITTI_RAW_DATA}/${kitti_day}/${sequence}_sync/image_02/data" "${category}/${sequence}/image_02/data"
		fi
		if [ ! -e "${category}/${sequence}/image_03/data" ]; then
			ln -s "${KITTI_RAW_DATA}/${kitti_day}/${sequence}_sync/image_03/data" "${category}/${sequence}/image_03/data"
                fi
                if [ ! -e "${category}/${sequence}/calib/calib_cam_to_cam.txt" ]; then
			ln -s "${KITTI_RAW_DATA}/${kitti_day}/calib_cam_to_cam.txt" "${category}/${sequence}/calib/calib_cam_to_cam.txt"
                fi
                if [ ! -e "${category}/${sequence}/calib/calib_velo_to_cam.txt" ]; then
                        ln -s "${KITTI_RAW_DATA}/${kitti_day}/calib_velo_to_cam.txt" "${category}/${sequence}/calib/calib_velo_to_cam.txt"
                fi
                if [ ! -e "${category}/${sequence}/calib/calib_imu_to_velo.txt" ]; then
                        ln -s "${KITTI_RAW_DATA}/${kitti_day}/calib_imu_to_velo.txt" "${category}/${sequence}/calib/calib_imu_to_velo.txt"
                fi
                if [ ! -e "${category}/${sequence}/velodyne_points" ]; then
                        ln -s "${KITTI_RAW_DATA}/${kitti_day}/${sequence}_sync/velodyne_points" "${category}/${sequence}/velodyne_points"
                fi
                if [ ! -e "${category}/${sequence}/oxts" ]; then
                        ln -s "${KITTI_RAW_DATA}/${kitti_day}/${sequence}_sync/oxts" "${category}/${sequence}/oxts"
                fi
	done
done
