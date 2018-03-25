import os, os.path
import random
import scipy.io as sio
import numpy as np
import lmdb
from shutil import copyfile
import cv2
import json
import argparse

import sys
from os.path import expanduser
home = expanduser("~")
caffe_root = home + '/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe

class kittiEigenBuilder():
    def __init__(self):
        self.train_scenes = [
                         'residential/2011_09_30_drive_0033',
                         'residential/2011_09_26_drive_0087',
                         'residential/2011_09_30_drive_0020',
                         'residential/2011_09_26_drive_0039',
                         'residential/2011_09_30_drive_0028',
                         'city/2011_09_26_drive_0018',
                         'residential/2011_09_26_drive_0035',
                         'city/2011_09_26_drive_0057',
                         'road/2011_10_03_drive_0042',
                         'residential/2011_09_26_drive_0022',
                         'road/2011_09_26_drive_0028',
                         'residential/2011_10_03_drive_0034',
                         'road/2011_09_29_drive_0004',
                         'road/2011_09_26_drive_0070',
                         'residential/2011_09_26_drive_0061',
                         'city/2011_09_26_drive_0091',
                         'city/2011_09_29_drive_0026',
                         'city/2011_09_26_drive_0014',
                         'city/2011_09_26_drive_0104',
                         'city/2011_09_26_drive_0001',
                         'city/2011_09_26_drive_0017',
                         'city/2011_09_26_drive_0051',
                         'residential/2011_09_30_drive_0034',
                         'city/2011_09_26_drive_0095',
                         'city/2011_09_26_drive_0060',
                         'residential/2011_09_26_drive_0079',
                         'road/2011_09_26_drive_0015',
                         'residential/2011_09_26_drive_0019',
                         'city/2011_09_26_drive_0005',
                         'city/2011_09_26_drive_0011',
                         'road/2011_09_26_drive_0032',
                         'city/2011_09_28_drive_0001',
                         'city/2011_09_26_drive_0113']




    def setup(self,setup_opt):    
        self.train_frame_distance = setup_opt['train_frame_distance']
        self.raw_data_dir = setup_opt['raw_data_dir']
        self.dataset_dir = setup_opt['dataset_dir']
        self.image_size = setup_opt['image_size']

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)


    def getData(self,isTrain):
        # ----------------------------------------------------------------------
        # Get dataset: image pairs (L1,L2,R1,R2 & K &T(L2R))
        # ----------------------------------------------------------------------
        self.L1_set = []
        self.R1_set = []
        self.L2_set = []
        self.R2_set = []
        self.K = []
        self.T = []

        scenes = self.train_scenes

        for cnt, scene in enumerate(scenes):
            print "Getting data. [Scene: ", cnt,  "/" , len(scenes), "]"
            seq_path = "/".join([self.raw_data_dir, scene,"image_02", "data"])
            seq_end = len(os.listdir(seq_path))-1
            for i in xrange(0, seq_end - self.train_frame_distance + 1):
                L1 = "/".join([self.raw_data_dir, scene, "image_02", "data", '{:010}'.format(i)]) + ".png"
                L2 = "/".join([self.raw_data_dir, scene, "image_02", "data", '{:010}'.format(i+self.train_frame_distance)]) + ".png"
                R1 = "/".join([self.raw_data_dir, scene, "image_03", "data", '{:010}'.format(i)]) + ".png"
                R2 = "/".join([self.raw_data_dir, scene, "image_03", "data", '{:010}'.format(i+self.train_frame_distance)]) + ".png"

                self.L1_set.append(L1)
                self.L2_set.append(L2)
                self.R1_set.append(R1)
                self.R2_set.append(R2)

                kt_scene =  "/".join([self.raw_data_dir, scene])

                KT = self.getKT(kt_scene) #Get K and T(right-to-left)
                self.K.append(KT[:4])
                self.T.append(KT[4])

    def getKT(self,scene):
        # ----------------------------------------------------------------------
        # Get K (camera intrinsic) and T (camera extrinsic)
        # ----------------------------------------------------------------------
        new_image_size = [float(self.image_size[0]), float(self.image_size[1])] #[height,width]

        # ----------------------------------------------------------------------
        # Get original K
        # ----------------------------------------------------------------------
        f = open(scene+"/calib/calib_cam_to_cam.txt", 'r')
        camTxt = f.readlines()
        f.close()
        K_dict = {}
        for line in camTxt:
            line_split = line.split(":")
            K_dict[line_split[0]] = line_split[1]

        # ----------------------------------------------------------------------
        # original K02
        # ----------------------------------------------------------------------
        P_split = K_dict["P_rect_02"].split(" ")
        S_split = K_dict["S_rect_02"].split(" ")
        ref_img_size = [float(S_split[2]), float(S_split[1])] # height, width


        # ----------------------------------------------------------------------
        # Get new K & position
        # ----------------------------------------------------------------------
        W_ratio = new_image_size[1] / ref_img_size[1]
        H_ratio = new_image_size[0] / ref_img_size[0]
        fx = float(P_split[1]) * W_ratio
        fy = float(P_split[6]) * H_ratio
        cx = float(P_split[3]) * W_ratio
        cy = float(P_split[7]) * H_ratio

        tx_L = float(P_split[4]) / float(P_split[1])
        # ty_L = float(P_split[8]) / float(P_split[6])

        # ----------------------------------------------------------------------
        # original K03
        # ----------------------------------------------------------------------
        P_split = K_dict["P_rect_03"].split(" ")
        S_split = K_dict["S_rect_03"].split(" ")

        tx_R = float(P_split[4]) / float(P_split[1])
        # ty_R = float(P_split[8]) / float(P_split[6])

        # ----------------------------------------------------------------------
        # Get position of Right camera w.r.t Left
        # ----------------------------------------------------------------------
        Tx = np.abs(tx_R - tx_L)
        # Ty = np.abs(tx_R - tx_L)

        se3 = [0,0,0,Tx,0,0]

        return [fx,fy,cx,cy,se3]



    def shuffleDataset(self):
        list_ = list(zip(self.L1_set, self.L2_set, self.R1_set, self.R2_set, self.K, self.T))
        random.shuffle(list_)
        self.L1_set, self.L2_set, self.R1_set, self.R2_set, self.K, self.T = zip(*list_)

    def saveDataset(self, isTrain, with_val):
        if isTrain:
            start_idx = 0
            end_idx = len(self.L1_set) 
            if with_val:
                end_idx = 22600

            txt_to_save = "/".join([self.dataset_dir,"train_left_1.txt"])
            self.saveTxt(txt_to_save, self.L1_set[start_idx:end_idx])

            txt_to_save = "/".join([self.dataset_dir,"train_right_1.txt"])
            self.saveTxt(txt_to_save, self.R1_set[start_idx:end_idx])

            txt_to_save = "/".join([self.dataset_dir,"train_left_2.txt"])
            self.saveTxt(txt_to_save, self.L2_set[start_idx:end_idx])

            txt_to_save = "/".join([self.dataset_dir,"train_right_2.txt"])
            self.saveTxt(txt_to_save, self.R2_set[start_idx:end_idx])

            lmdb_to_save = "/".join([self.dataset_dir,"train_K"])
            self.saveLmdb(lmdb_to_save, np.expand_dims(np.expand_dims(np.asarray(self.K[start_idx:end_idx]),3),4))

            lmdb_to_save = "/".join([self.dataset_dir,"train_T_R2L"])
            self.saveLmdb(lmdb_to_save, np.expand_dims(np.expand_dims(np.asarray(self.T[start_idx:end_idx]),3),4))
            print "Dataset built! Number of training instances: ", len(self.L1_set[start_idx:end_idx])
        else:
            start_idx = 22600
            txt_to_save = "/".join([self.dataset_dir,"val_left_1.txt"])
            self.saveTxt(txt_to_save, self.L1_set[start_idx:])

            txt_to_save = "/".join([self.dataset_dir,"val_right_1.txt"])
            self.saveTxt(txt_to_save, self.R1_set[start_idx:])

            txt_to_save = "/".join([self.dataset_dir,"val_left_2.txt"])
            self.saveTxt(txt_to_save, self.L2_set[start_idx:])

            txt_to_save = "/".join([self.dataset_dir,"val_right_2.txt"])
            self.saveTxt(txt_to_save, self.R2_set[start_idx:])

            lmdb_to_save = "/".join([self.dataset_dir,"val_K"])
            self.saveLmdb(lmdb_to_save, np.expand_dims(np.expand_dims(np.asarray(self.K[start_idx:]),3),4))

            lmdb_to_save = "/".join([self.dataset_dir,"val_T_R2L"])
            self.saveLmdb(lmdb_to_save, np.expand_dims(np.expand_dims(np.asarray(self.T[start_idx:]),3),4))

            print "Dataset built! Number of validation instances: ", len(self.L1_set[start_idx:])


    def saveTxt(self, path, img_list):
        f = open(path, 'w')
        for line in img_list:
            f.writelines(line+"\n")
        f.close()


    def saveLmdb(self, path, np_arr):
        # input: np_arr: shape = (N,C,H,W)
        N = np_arr.shape[0]
        map_size = np_arr.nbytes * 10
        env = lmdb.open(path, map_size=map_size)

        with env.begin(write=True) as txn:
            # txn is a Transaction object
            for i in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = np_arr.shape[1]
                datum.height = np_arr.shape[2]
                datum.width = np_arr.shape[3]
                datum = caffe.io.array_to_datum(np_arr[i])
                str_id = '{:08}'.format(i)
                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())



parser = argparse.ArgumentParser(description='Dataset builder')
parser.add_argument('--builder', type=str, default='kitti_eigen', help='Select builder (kitti_eigen; kitti_dometry)')
parser.add_argument('--train_frame_distance', type=int, default=1, help='Frame distance between training instances')
parser.add_argument('--raw_data_dir', type=str, default='./data/kitti_raw_data', help='Directory path storing the raw KITTI dataset')
parser.add_argument('--dataset_dir', type=str, default='./data/dataset/kitti_eigen', help='Directory path storing the created dataset')
parser.add_argument('--image_size', type=list, default=[160, 608], help='Image size for the dataset [height, width]')
parser.add_argument('--with_val', type=bool, default=False, help='Building validation set as well')

args = parser.parse_args()
args.image_size = [int("".join(args.image_size).split(",")[0][1:]), int("".join(args.image_size).split(",")[1][:-1])]

if args.builder == "kitti_eigen":
    builder = kittiEigenBuilder()

    # ----------------------------------------------------------------------
    # Setup options
    # ----------------------------------------------------------------------
    setup_opt = {}
    setup_opt['train_frame_distance'] = args.train_frame_distance
    setup_opt['raw_data_dir'] = args.raw_data_dir
    setup_opt['dataset_dir'] = args.dataset_dir
    setup_opt['image_size'] = args.image_size

    builder.setup(setup_opt)
    # ----------------------------------------------------------------------
    # Training set
    # ----------------------------------------------------------------------
    builder.getData(isTrain=True)
    builder.shuffleDataset()
    builder.saveDataset(isTrain=True, with_val=args.with_val)
    # ----------------------------------------------------------------------
    # Validation set
    # ----------------------------------------------------------------------
    if args.with_val:
        builder.saveDataset(isTrain=False)  


