import copy
from glob import glob
import numpy as np
import os

from pose_evaluation_utils import *

gt_dir = "./sfmLearner/ground_truth/"
pred_dir = "./sfmLearner/ours_results/"
solve_global_pose = False


def load_pose_from_list(traj_list, idx):
    """Load pose from SfM-learner list
    Args:
        traj_list (list): snippet trajectory list
        idx (int): index
    Returns:
        pose_mat (4x4 array): pose array
    """
    pose = traj_list[list(traj_list.keys())[idx]]
    pose = [float(i) for i in pose]
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = quat2mat([pose[6], pose[3], pose[4], pose[5]])
    pose_mat[:3, 3] = pose[:3]
    return pose_mat


def save_traj(txt, poses):
    """Save trajectory (absolute poses) as KITTI odometry file format
    Args:
        txt (str): pose text file path
        poses (array dict): poses, each pose is 4x4 array
    """
    with open(txt, "w") as f:
        for i in poses:
            pose = poses[i]
            pose = pose.flatten()[:12]
            line_to_write = " ".join([str(i) for i in pose])
            f.writelines(line_to_write+"\n")
    print("Trajectory saved.")


for seq in ['09', '10']:
    gt_files = sorted(glob(os.path.join(gt_dir, seq, "*.txt")))
    pred_files = sorted(glob(os.path.join(pred_dir, seq, "*.txt")))

    for pose_type in ['pred', 'gt']:
        poses = {0: np.eye(4)}
        for cnt in range(len(gt_files)):
            scale = 1
            if not(solve_global_pose):
                # Solve pose scale
                ate, scale = compute_ate(gt_files[cnt], pred_files[cnt])

            # Read pred pose
            if pose_type == "pred":
                traj_list = read_file_list(pred_files[cnt])
            elif pose_type == "gt":
                traj_list = read_file_list(gt_files[cnt])

            if cnt < len(gt_files) - 1:
                # Read second pose in the traj file
                poses[cnt+1] = load_pose_from_list(traj_list, 1)
                poses[cnt+1][:3, 3] *= scale
                # Transform the pose w.r.t first frame
                poses[cnt+1] = poses[cnt] @ poses[cnt+1]
            else:
                # Read second to last poses
                for k in range(1, len(traj_list)):
                    poses[cnt+k] = load_pose_from_list(traj_list, k)
                    poses[cnt+k][:3, 3] *= scale
                    poses[cnt+k] = poses[cnt+k-1] @ poses[cnt+k]

        if pose_type == "pred":
            poses_pred = copy.deepcopy(poses)
        elif pose_type == "gt":
            poses_gt = copy.deepcopy(poses)

    # If solve global pose
    if solve_global_pose:
        # Read XYZ
        gtruth_xyz = []
        pred_xyz = []
        for cnt in poses:
            gtruth_xyz.append(poses_gt[cnt][:3, 3])
            pred_xyz.append(poses_pred[cnt][:3, 3])
        gtruth_xyz = np.asarray(gtruth_xyz)
        pred_xyz = np.asarray(pred_xyz)

        # Solve for global scale
        scale = np.sum(gtruth_xyz * pred_xyz)/np.sum(pred_xyz ** 2)

        # Update pose
        for cnt in poses:
            poses_pred[cnt][:3, 3] *= scale

    save_traj("./{}.txt".format(seq), poses_pred)
