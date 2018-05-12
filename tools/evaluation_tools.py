#!/usr/bin/env python
import numpy as np
import sys
import numpy as np
from matplotlib import pyplot as plt

caffe_root = '$YOUR_CAFFE_DIR'  
sys.path.insert(0, caffe_root + 'python')
import caffe

import h5py
import os, os.path
import cv2
import argparse

parser = argparse.ArgumentParser(description='Evaluation toolkit')
parser.add_argument('--func', type=str, default='generate_depth_npy', help='Select function (generate_depth_npy; generate_odom_result; eval_odom)')
parser.add_argument('--dataset', type=str, default='kitti_eigen', help='Select dataset (kitti_eigen)')
parser.add_argument('--model', type=str, help='Depth caffemodel')


parser.add_argument('--depth_net_def', type=str, default="experiments/networks/depth_deploy.prototxt", help='Depth network prototxt')
parser.add_argument('--npy_dir', type=str, default='./result/depth/depths', help='Directory path storing the created npy file')

parser.add_argument('--odom_net_def', type=str, default="experiments/networks/odometry_deploy.prototxt", help='Visual odometry network prototxt')
parser.add_argument('--odom_result_dir', type=str, default='./result/depth_odometry/odom_result', help='Directory path storing the odometry results')


global args 
args = parser.parse_args()

# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)

class kittiEigenGenerateDepthNpy():
		def __init__(self):
			depth_net_def = args.depth_net_def
			caffe_model = args.model
			self.depth_net = caffe.Net(depth_net_def, caffe_model, caffe.TEST)
			self.image_width = self.depth_net.blobs['img'].data.shape[3]
			self.image_height = self.depth_net.blobs['img'].data.shape[2]

			# ----------------------------------------------------------------------
			# Check evaluation set exist
			# ----------------------------------------------------------------------
			self.dataset_path = "./data/depth_evaluation/kitti_eigen"
			assert(os.path.exists(self.dataset_path)==True)

		def getImage(self, img_path): 
			# ----------------------------------------------------------------------
			# Get and preprocess image
			# ----------------------------------------------------------------------
			img = cv2.imread(img_path)
			if img==None:
				print "img_path: ", img_path
				assert img!=None, "Image reading error. Check whether your image path is correct or not."
			img = cv2.resize(img, (self.image_width, self.image_height))
			img = img.transpose((2,0,1))
			img = img.astype(np.float32)
			img[0] -= 104
			img[1] -= 117
			img[2] -= 123
			return img

		def getPredInvDepths(self):
			inv_depths = []
			for cnt in xrange(697):
				print "Getting prediction: ", cnt, " / 697" 
				img_path = self.dataset_path + "/left_rgb/" + str(cnt) + ".png"
				img = self.getImage(img_path)
				self.depth_net.blobs['img'].data[0] = img #dimension (3,H,W)
				self.depth_net.forward();
				inv_depths.append(self.depth_net.blobs["inv_depth"].data[0,0].copy())
			inv_depths = np.asarray(inv_depths)
			return inv_depths

		def saveNpy(self, inv_depths):
			npy_folder_dir = '/'.join(args.npy_dir.split('/')[:-1])
			if not os.path.exists(npy_folder_dir):
				os.makedirs(npy_folder_dir)
			np.save(args.npy_dir, inv_depths)

class kittiPredOdom():
	def __init__(self):
		model_def = args.odom_net_def
		caffe_model = args.model
		self.odom_net = caffe.Net(model_def, caffe_model, caffe.TEST)
		self.image_width = self.odom_net.blobs['imgs'].data.shape[3]
		self.image_height = self.odom_net.blobs['imgs'].data.shape[2]

		self.result_path = args.odom_result_dir

		self.eval_seqs = ["00", "01", "02", "04", "05", "06", "07", "08", "09", "10"]
		self.eval_seqs_start_end = {
									"00": [0, 4540], 
									"01": [0, 1100], 
									"02": [0, 4660], 
									"04": [0, 270], 
									"05": [0, 2760],
									"06": [0, 1100], 
									"07": [0, 1100], 
									"08": [1100, 5170], 
									"09": [0, 1590], 
									"10": [0, 1200]
									}

		self.eval_seqs_path		 = {
									"00": "residential/2011_10_03_drive_0027", 
									"01": "road/2011_10_03_drive_0042",
									"02": "residential/2011_10_03_drive_0034", 
									"04": "road/2011_09_30_drive_0016", 
									"05": "residential/2011_09_30_drive_0018",
									"06": "residential/2011_09_30_drive_0020", 
									"07": "residential/2011_09_30_drive_0027", 
									"08": "residential/2011_09_30_drive_0028", 
									"09": "residential/2011_09_30_drive_0033", 
									"10": "residential/2011_09_30_drive_0034"
									}


	def getImage(self, img_path): 
		# ----------------------------------------------------------------------
		# Get and preprocess image
		# ----------------------------------------------------------------------
		img = cv2.imread(img_path)
		if img==None:
				print "img_path: ", img_path
				assert img!=None, "Image reading error. Check whether your image path is correct or not."
		img = cv2.resize(img, (self.image_width, self.image_height))
		img = img.transpose((2,0,1))
		img = img.astype(np.float32)
		img[0] -= 104
		img[1] -= 117
		img[2] -= 123
		return img

	def getPredInvDepths(self):
		inv_depths = []
		for cnt in xrange(697):
			img_path = self.dataset_path + "/left_rgb/" + str(cnt) + ".png"
			img = self.getImage(img_path)
			self.depth_net.blobs['img'].data[0] = img #dimension (3,H,W)
			self.depth_net.forward();
			inv_depths.append(self.depth_net.blobs["inv_depth"].data[0,0].copy())
		inv_depths = np.asarray(inv_depths)
		return inv_depths

	def getPredPoses(self):
		pred_poses = {}
		for cnt,seq in enumerate(self.eval_seqs):
			print "Getting predictions... Sequence: ", cnt, " / ",len(self.eval_seqs) 
			pred_poses[seq] = []
			seq_path = "./data/kitti_raw_data/" + self.eval_seqs_path[seq]
			start_idx = self.eval_seqs_start_end[seq][0]
			end_idx = self.eval_seqs_start_end[seq][1]
			for idx in xrange(start_idx, end_idx):
				img1_path = seq_path + "/image_02/data/{:010}.png".format(idx)
				img2_path = seq_path + "/image_02/data/{:010}.png".format(idx+1)
				img1 = self.getImage(img1_path)
				img2 = self.getImage(img2_path)
				self.odom_net.blobs['imgs'].data[0,:3] = img2
				self.odom_net.blobs['imgs'].data[0,3:] = img1
				self.odom_net.forward();
				pred_poses[seq].append(self.odom_net.blobs['SE3'].data[0,0].copy())
		return pred_poses

	def SE3_cam2world(self, pred_poses):
		self.pred_SE3_world = {}
		for seq in self.eval_seqs:
			cur_T = np.eye(4)
			tmp_SE3_world = []
			tmp_SE3_world.append(cur_T)
			for pose in pred_poses[seq]:
				cur_T = np.dot(cur_T, pose)
				tmp_SE3_world.append(cur_T)
			self.pred_SE3_world[seq] = tmp_SE3_world

	def saveResultPoses(self):
		result_dir = args.odom_result_dir
		if not os.path.exists(self.result_path):
			os.makedirs(self.result_path)

		for seq in self.eval_seqs:
			f = open(self.result_path + "/" + seq + ".txt", 'w')
			for cnt, SE3 in enumerate(self.pred_SE3_world[seq]):
				tx = str(SE3[0,3])
				ty = str(SE3[1,3])
				tz = str(SE3[2,3])
				R00 = str(SE3[0,0])
				R01 = str(SE3[0,1])
				R02 = str(SE3[0,2])
				R10 = str(SE3[1,0])
				R11 = str(SE3[1,1])
				R12 = str(SE3[1,2])
				R20 = str(SE3[2,0])
				R21 = str(SE3[2,1])
				R22 = str(SE3[2,2])
				line_to_write = " ".join([R00, R01, R02, tx, R10, R11, R12, ty, R20, R21, R22, tz])
				f.writelines(line_to_write+"\n")
			f.close()

	def rot2quat(self,R):
	    rz, ry, rx = self.mat2euler(R)
	    qw, qx, qy, qz = self.euler2quat(rz, ry, rx)
	    return qw, qx, qy, qz

	def quat2mat(self,q):
	    ''' Calculate rotation matrix corresponding to quaternion
	    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
	    Parameters
	    ----------
	    q : 4 element array-like

	    Returns
	    -------
	    M : (3,3) array
	      Rotation matrix corresponding to input quaternion *q*

	    Notes
	    -----
	    Rotation matrix applies to column vectors, and is applied to the
	    left of coordinate vectors.  The algorithm here allows non-unit
	    quaternions.

	    References
	    ----------
	    Algorithm from
	    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

	    Examples
	    --------
	    >>> import numpy as np
	    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
	    >>> np.allclose(M, np.eye(3))
	    True
	    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
	    >>> np.allclose(M, np.diag([1, -1, -1]))
	    True
	    '''
	    w, x, y, z = q
	    Nq = w*w + x*x + y*y + z*z
	    if Nq < 1e-8:
	        return np.eye(3)
	    s = 2.0/Nq
	    X = x*s
	    Y = y*s
	    Z = z*s
	    wX = w*X; wY = w*Y; wZ = w*Z
	    xX = x*X; xY = x*Y; xZ = x*Z
	    yY = y*Y; yZ = y*Z; zZ = z*Z
	    return np.array(
	           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
	            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
	            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

	def mat2euler(self,M, cy_thresh=None, seq='zyx'):
	    '''
	    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
	    Discover Euler angle vector from 3x3 matrix
	    Uses the conventions above.
	    Parameters
	    ----------
	    M : array-like, shape (3,3)
	    cy_thresh : None or scalar, optional
	     threshold below which to give up on straightforward arctan for
	     estimating x rotation.  If None (default), estimate from
	     precision of input.
	    Returns
	    -------
	    z : scalar
	    y : scalar
	    x : scalar
	     Rotations in radians around z, y, x axes, respectively
	    Notes
	    -----
	    If there was no numerical error, the routine could be derived using
	    Sympy expression for z then y then x rotation matrix, which is::
	    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
	    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
	    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
	    with the obvious derivations for z, y, and x
	     z = atan2(-r12, r11)
	     y = asin(r13)
	     x = atan2(-r23, r33)
	    for x,y,z order
	    y = asin(-r31)
	    x = atan2(r32, r33)
	    z = atan2(r21, r11)
	    Problems arise when cos(y) is close to zero, because both of::
	     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
	     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
	    will be close to atan2(0, 0), and highly unstable.
	    The ``cy`` fix for numerical instability below is from: *Graphics
	    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
	    0123361559.  Specifically it comes from EulerAngles.c by Ken
	    Shoemake, and deals with the case where cos(y) is close to zero:
	    See: http://www.graphicsgems.org/
	    The code appears to be licensed (from the website) as "can be used
	    without restrictions".
	    '''
	    M = np.asarray(M)
	    if cy_thresh is None:
	        try:
	            cy_thresh = np.finfo(M.dtype).eps * 4
	        except ValueError:
	            cy_thresh = _FLOAT_EPS_4
	    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
	    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
	    cy = math.sqrt(r33*r33 + r23*r23)
	    if seq=='zyx':
	        if cy > cy_thresh: # cos(y) not close to zero, standard form
	            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
	            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
	            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
	        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
	            # so r21 -> sin(z), r22 -> cos(z) and
	            z = math.atan2(r21,  r22)
	            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
	            x = 0.0
	    elif seq=='xyz':
	        if cy > cy_thresh:
	            y = math.atan2(-r31, cy)
	            x = math.atan2(r32, r33)
	            z = math.atan2(r21, r11)
	        else:
	            z = 0.0
	            if r31 < 0:
	                y = np.pi/2
	                x = atan2(r12, r13)
	            else:
	                y = -np.pi/2
	    else:
	        raise Exception('Sequence not recognized')
	    return z, y, x

	def euler2quat(self,z=0, y=0, x=0, isRadian=True):
	    ''' Return quaternion corresponding to these Euler angles
	    Uses the z, then y, then x convention above
	    Parameters
	    ----------
	    z : scalar
	         Rotation angle in radians around z-axis (performed first)
	    y : scalar
	         Rotation angle in radians around y-axis
	    x : scalar
	         Rotation angle in radians around x-axis (performed last)
	    Returns
	    -------
	    quat : array shape (4,)
	         Quaternion in w, x, y z (real, then vector) format
	    Notes
	    -----
	    We can derive this formula in Sympy using:
	    1. Formula giving quaternion corresponding to rotation of theta radians
	         about arbitrary axis:
	         http://mathworld.wolfram.com/EulerParameters.html
	    2. Generated formulae from 1.) for quaternions corresponding to
	         theta radians rotations about ``x, y, z`` axes
	    3. Apply quaternion multiplication formula -
	         http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
	         formulae from 2.) to give formula for combined rotations.
	    '''
	  
	    if not isRadian:
	        z = ((np.pi)/180.) * z
	        y = ((np.pi)/180.) * y
	        x = ((np.pi)/180.) * x
	    z = z/2.0
	    y = y/2.0
	    x = x/2.0
	    cz = math.cos(z)
	    sz = math.sin(z)
	    cy = math.cos(y)
	    sy = math.sin(y)
	    cx = math.cos(x)
	    sx = math.sin(x)
	    return np.array([
	                     cx*cy*cz - sx*sy*sz,
	                     cx*sy*sz + cy*cz*sx,
	                     cx*cz*sy - sx*cy*sz,
	                     cx*cy*sz + sx*cz*sy])
	
class kittiEvalOdom():
	# ----------------------------------------------------------------------
	# poses: N,4,4
	# pose: 4,4
	# ----------------------------------------------------------------------
	def __init__(self):
		self.lengths= [100,200,300,400,500,600,700,800]
		self.num_lengths = len(self.lengths)
		self.gt_dir = "./data/odometry_evaluation/poses"

	def loadPoses(self, file_name):
		# ----------------------------------------------------------------------
		# Each line in the file should follow one of the following structures
		# (1) idx pose(3x4 matrix in terms of 12 numbers)
		# (2) pose(3x4 matrix in terms of 12 numbers)
		# ----------------------------------------------------------------------
		f = open(file_name, 'r')
		s = f.readlines()
		f.close()
		file_len = len(s)
		poses = {}
		for cnt, line in enumerate(s):
			P = np.eye(4)
			line_split = [float(i) for i in line.split(" ")]
			withIdx = int(len(line_split)==13)
			for row in xrange(3):
				for col in xrange(4):
					P[row, col] = line_split[row*4+col+ withIdx]
			if withIdx:
				frame_idx = line_split[0]
			else:
				frame_idx = cnt
			poses[frame_idx] = P
		return poses

	def trajectoryDistances(self, poses):
		# ----------------------------------------------------------------------
		# poses: dictionary: [frame_idx: pose]
		# ----------------------------------------------------------------------
		dist = [0]
		sort_frame_idx = sorted(poses.keys())
		for i in xrange(len(sort_frame_idx)-1):
			cur_frame_idx = sort_frame_idx[i]
			next_frame_idx = sort_frame_idx[i+1]
			P1 = poses[cur_frame_idx]
			P2 = poses[next_frame_idx]
			dx = P1[0,3] - P2[0,3]
			dy = P1[1,3] - P2[1,3]
			dz = P1[2,3] - P2[2,3]
			dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))	
		return dist

	def rotationError(self, pose_error):
		a = pose_error[0,0]
		b = pose_error[1,1]
		c = pose_error[2,2]
		d = 0.5*(a+b+c-1.0)
		return np.arccos(max(min(d,1.0),-1.0))

	def translationError(self, pose_error):
		dx = pose_error[0,3]
		dy = pose_error[1,3]
		dz = pose_error[2,3]
		return np.sqrt(dx**2+dy**2+dz**2)

	def lastFrameFromSegmentLength(self, dist, first_frame, len_):
		for i in xrange(first_frame, len(dist), 1):
			if dist[i] > (dist[first_frame] + len_):
				return i
		return -1

	def calcSequenceErrors(self, poses_gt, poses_result):
		err = []
		dist = self.trajectoryDistances(poses_gt)
		self.step_size = 10
		
		for first_frame in xrange(9, len(poses_gt), self.step_size):
			for i in xrange(self.num_lengths):
				len_ = self.lengths[i]
				last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

				# ----------------------------------------------------------------------
				# Continue if sequence not long enough
				# ----------------------------------------------------------------------
				if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
					continue

				# ----------------------------------------------------------------------
				# compute rotational and translational errors
				# ----------------------------------------------------------------------
				pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
				pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
				pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

				r_err = self.rotationError(pose_error)
				t_err = self.translationError(pose_error)

				# ----------------------------------------------------------------------
				# compute speed 
				# ----------------------------------------------------------------------
				num_frames = last_frame - first_frame + 1.0
				speed = len_/(0.1*num_frames)

				err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
		return err
		
	def saveSequenceErrors(self, err, file_name):
		fp = open(file_name,'w')
		for i in err:
			line_to_write = " ".join([str(j) for j in i])
			fp.writelines(line_to_write+"\n")
		fp.close()

	def computeOverallErr(self, seq_err):
		t_err = 0
		r_err = 0

		seq_len = len(seq_err)

		for item in seq_err:
			r_err += item[1]
			t_err += item[2]
		ave_t_err = t_err / seq_len
		ave_r_err = r_err / seq_len
		return ave_t_err, ave_r_err 

	def plotPath(self, seq, poses_gt, poses_result):
		plot_keys = ["Ground Truth", "Ours"]
		fontsize_ = 20
		plot_num =-1
			
		poses_dict = {}
		poses_dict["Ground Truth"] = poses_gt
		poses_dict["Ours"] = poses_result

		fig = plt.figure()
		ax = plt.gca()
		ax.set_aspect('equal')

		for key in plot_keys:
			pos_xz = []
			# for pose in poses_dict[key]:
			for frame_idx in sorted(poses_dict[key].keys()):
				pose = poses_dict[key][frame_idx]
				pos_xz.append([pose[0,3], pose[2,3]])
			pos_xz = np.asarray(pos_xz)
			plt.plot(pos_xz[:,0], pos_xz[:,1], label = key)	
			
		plt.legend(loc = "upper right", prop={'size': fontsize_})
		plt.xticks(fontsize = fontsize_) 
		plt.yticks(fontsize = fontsize_) 
		plt.xlabel('x (m)',fontsize = fontsize_)
		plt.ylabel('z (m)',fontsize = fontsize_)
		fig.set_size_inches(10, 10)
		png_title = "sequence_{:02}".format(seq)
		plt.savefig(self.plot_path_dir +  "/" + png_title + ".pdf",bbox_inches='tight', pad_inches=0)
		# plt.show()

	def plotError(self, avg_segment_errs):
		# ----------------------------------------------------------------------
		# avg_segment_errs: dict [100: err, 200: err...]
		# ----------------------------------------------------------------------
		plot_y = []
		plot_x = []
		for len_ in self.lengths:
			plot_x.append(len_)
			plot_y.append(avg_segment_errs[len_][0])
		fig = plt.figure()
		plt.plot(plot_x, plot_y)
		plt.show()

	def computeSegmentErr(self, seq_errs):
		# ----------------------------------------------------------------------
		# This function calculates average errors for different segment.
		# ----------------------------------------------------------------------

		segment_errs = {}
		avg_segment_errs = {}
		for len_ in self.lengths:
			segment_errs[len_] = []
		# ----------------------------------------------------------------------
		# Get errors
		# ----------------------------------------------------------------------
		for err in seq_errs:
			len_ = err[3]
			t_err = err[2]
			r_err = err[1]
			segment_errs[len_].append([t_err, r_err])
		# ----------------------------------------------------------------------
		# Compute average
		# ----------------------------------------------------------------------
		for len_ in self.lengths:
			if segment_errs[len_] != []:
				avg_t_err = np.mean(np.asarray(segment_errs[len_])[:,0])
				avg_r_err = np.mean(np.asarray(segment_errs[len_])[:,1])
				avg_segment_errs[len_] = [avg_t_err, avg_r_err]
			else:
				avg_segment_errs[len_] = []
		return avg_segment_errs

	def eval(self, result_dir):
		error_dir = result_dir + "/errors"
		self.plot_path_dir = result_dir + "/plot_path"
		plot_error_dir = result_dir + "/plot_error"

		if not os.path.exists(error_dir):
			os.makedirs(error_dir)
		if not os.path.exists(self.plot_path_dir):
			os.makedirs(self.plot_path_dir)
		if not os.path.exists(plot_error_dir):
			os.makedirs(plot_error_dir)

		total_err = []

		ave_t_errs = []
		ave_r_errs = []

		for i in self.eval_seqs:
			self.cur_seq = '{:02}'.format(i)
			file_name = '{:02}.txt'.format(i)

			poses_result = self.loadPoses(result_dir+"/"+file_name)
			poses_gt = self.loadPoses(self.gt_dir + "/" + file_name)
			self.result_file_name = result_dir+file_name

			# ----------------------------------------------------------------------
			# compute sequence errors
			# ----------------------------------------------------------------------
			seq_err = self.calcSequenceErrors(poses_gt, poses_result)
			self.saveSequenceErrors(seq_err, error_dir + "/" + file_name)

			# ----------------------------------------------------------------------
			# Compute segment errors
			# ----------------------------------------------------------------------
			avg_segment_errs = self.computeSegmentErr(seq_err)

			# ----------------------------------------------------------------------
			# compute overall error
			# ----------------------------------------------------------------------
			ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
			print "Sequence: " + str(i)
			print "Average translational RMSE (%): ", ave_t_err*100
			print "Average rotational error (deg/100m): ", ave_r_err/np.pi * 180 *100
			ave_t_errs.append(ave_t_err)
			ave_r_errs.append(ave_r_err)

			# ----------------------------------------------------------------------
			# Ploting (To-do)
			# (1) plot trajectory
			# (2) plot per segment error
			# ----------------------------------------------------------------------
			self.plotPath(i,poses_gt, poses_result)
			# self.plotError(avg_segment_errs)

		print "-------------------- For Copying ------------------------------"
		for i in xrange(len(ave_t_errs)):
			print "{0:.2f}".format(ave_t_errs[i]*100)
			print "{0:.2f}".format(ave_r_errs[i]/np.pi*180*100)
		print "-------------------- For copying ------------------------------"

if args.func == "generate_depth_npy":
	if args.dataset == "kitti_eigen":
		generator = kittiEigenGenerateDepthNpy()
		inv_depths = generator.getPredInvDepths()
		generator.saveNpy(inv_depths)

elif args.func == "generate_odom_result":
	print "Getting predictions..."
	generator = kittiPredOdom()
	pred_poses = generator.getPredPoses()
	print "Converting to world coordinates..."
	generator.SE3_cam2world(pred_poses)
	print "Saving predictions..."
	generator.saveResultPoses()

elif args.func == "eval_odom":
	odom_eval = kittiEvalOdom()
	odom_eval.eval_seqs = [0,1,2,4,5,6,7,8,9,10] # Seq 03 is missing since the dataset is not available in KITTI homepage.
	odom_eval.eval(args.odom_result_dir)




