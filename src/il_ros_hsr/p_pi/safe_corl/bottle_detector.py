
import hsrb_interface
import rospy
import sys
import math
import tf
import tf2_ros
import tf2_geometry_msgs
import IPython
import cv2
import os
sys.path.append('../RCNN-Obj-Dectect')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped

import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython

from il_ros_hsr.core.sensors import  RGBD

sys.path.append('../Faster-RCNN_TF/lib/')
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from networks.factory import get_network
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy.linalg as LA

import tensorflow as tfl
import numpy as np
import time

from image_geometry import PinholeCameraModel as PCM
from itertools import combinations


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


class Bottle_Detect():

	def __init__(self,cam_info):
		
		self.label = 'bottle'
		

		self.poses = []


	
		self.count = 0
		self.sess = tfl.Session()

		self.pcm = PCM()
		self.pcm.fromCameraInfo(cam_info)
		# load network
		#IPython.embed()
		self.br = tf.TransformBroadcaster()
	
		self.net = get_network('VGGnet_test')
		# load model
		self.saver = tfl.train.Saver(write_version=tfl.train.SaverDef.V1)
		self.saver.restore(self.sess, '/home/autolab/Workspaces/michael_working/Faster-RCNN_TF/model/VGGnet_fast_rcnn_iter_70000.ckpt')



	def vis_detections(self,im, class_name, dets, thresh=0.5):
		"""Draw detected bounding boxes."""
		inds = np.where(dets[:, -1] >= thresh)[0]
		if len(inds) == 0:
			return [], None

		centers = []
		for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]
			im = im.copy()
			cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255))

			center = np.zeros(2)
			center[0] = bbox[0] + (bbox[2] - bbox[0])/2.0
			center[1] = bbox[1] + (bbox[3] - bbox[1])/2.0
			print center
			im[center[1]-20:center[1]+20,center[0]-20:center[0]+20,:] = [0,0,255]

			centers.append(center)
			
		return centers,im


	def detect(self,im):
		timer = Timer()
		timer.tic()
		scores, boxes = im_detect(self.sess, self.net, im)
		timer.toc()
		print ('Detection took {:.3f}s for '
		       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

		# Visualize detections for each class
		#im = im[:, :, (2, 1, 0)]

		# fig, ax = plt.subplots(figsize=(12, 12))
		# ax.imshow(im, aspect='equal')
		im_b = None
		centers = []

		CONF_THRESH = 0.5
		NMS_THRESH = 0.3
		for cls_ind, cls in enumerate(CLASSES[1:]):
			cls_ind += 1 # because we skipped background
			#IPython.embed()
			if(cls == self.label):
				cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
				cls_scores = scores[:, cls_ind]
				dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
				keep = nms(dets, NMS_THRESH)
				dets = dets[keep, :]

				centers,im_b = self.vis_detections(im, cls, dets, thresh=CONF_THRESH)
				

		return centers,im_b


	def get_image_box(self,im,center):
		WIDTH = 50
		HEIGHT = 50

		box = im[int(center[1])-WIDTH:int(center[1])+WIDTH,int(center[0])-HEIGHT:int(center[0])+HEIGHT,:]
		#box = cv2.cvtColor(box, cv2.cv.CV_RGB2GRAY)
		# cv2.imshow('box_making',box)
		# cv2.waitKey(300)
		# print "BOX SIZE"
		# print box.shape

		return box




	def get_depth(self,points,depth_img):
		#returns list of left (u,v,disparity)
		
		poses = []
		for p in points: 
			d_m = depth_img[p[1]-20:p[1]+20,p[0]-20:p[0]+20]
			indx = np.nonzero(d_m)
			try:
				lowest = np.amin(d_m[indx])
			except: 
				return poses
			
			if(lowest > 0):
				poses.append([p[0],p[1],0.001*lowest])

		return poses

	def detect_bottle(self,rgb_img,depth_img):

		found = False
		
		if(rgb_img == None):
			print rgb_img.shape
			rospy.logerr('no image from robot camera')
			return []
		

		p,rgb_img_b = self.detect(rgb_img)
		
		if(not rgb_img_b == None):
			found = True
			cv2.imshow('image_dect',rgb_img_b)
			cv2.waitKey(30)

		poses = self.get_depth(p,depth_img)
		p_poses = self.project_poses(poses)

		self.broadcast_poses(p_poses)

		return found,rgb_img_b,p_poses

	def project_poses(self,poses):
		project_poses = []
		for pose in poses:
			print "POSE ",pose
			#IPython.embed()
			td_points = self.pcm.projectPixelTo3dRay((pose[0],pose[1]))
			pose_p = np.array([td_points[0],td_points[1],pose[2]])
			project_poses.append(pose_p)	

		return project_poses


	def clean_up(self):
		self.sess.close()
		tfl.reset_default_graph()

	def broadcast_poses(self,poses):
		count = 0
		for pose in poses:
			print "POSE ",pose
			#IPython.embed()
			self.br.sendTransform((pose[0], pose[1]-0.05, pose[2]+0.06),
					(0.0, 0.0, 0.0, 1.0),
					rospy.Time.now(),
					'bottle'+str(count),
					'head_rgbd_sensor_link')
			count += 1





if __name__=='__main__':
	rospy.init_node('readJoy_node', anonymous=True)

	print'bottle'

	rgbd = RGBD()
	time.sleep(5)


	do = Bottle_Detect()
	while True: 
		print "IMGAE"
		print rgbd.read_color_data()
		if(not rgbd.read_color_data() == None):
			do.detect_bottle(rgbd.read_color_data())
