import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython
from numpy.random import normal
import time
#import listener
import thread

import glob
import os
import cPickle as pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
import rospy

import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')


import il_ros_hsr.p_pi.bed_making.config_bed as cfg


from detectors.grasp_detector import GDetector
from detectors.tran_detector import SDetector

rollouts = []


def load_held_out_rollouts():

	rollouts_p = glob.glob(os.path.join(cfg.STAT_PATH, '*_*'))


	for rollout_p in rollouts_p: 
		
		rollout = pickle.load(open(rollout_p+'/rollout.p'))

		rollouts.append(rollout)


def unscale(pose,img):

	w,h,d = img.shape
	
	pose[0] = pose[0]/w-0.5
	pose[1] = pose[1]/h-0.5

	return pose


def test_grasp():

	l2_grasp_score = []

	for rollout in rollouts:

		for d_point in rollout:

			if type(d_point) == list:
				continue
			
			if d_point['type'] == 'grasp':

				label = np.array(d_point['sup_pose'])

				label = unscale(label,d_point['c_img'])

				out_pred = d_point['net_pose']
				
				out_pred = unscale(out_pred,d_point['c_img'])

				l2_grasp_score.append(np.sum(np.square(label-out_pred)))

	return np.mean(l2_grasp_score)


def test_transistion():

	tran_score = []

	for rollout in rollouts:

		for d_point in rollout:

			if type(d_point) == list:
					continue
			
			if d_point['type'] == 'success':


				label = d_point['sup_class']

				out_pred = d_point['net_class']

			
				
				if (label == out_pred):
					tran_score.append(1.0)
				else:
					tran_score.append(0.0)



	return np.mean(tran_score)




if __name__ == "__main__":

	load_held_out_rollouts()

	print "GRASP SCORE ", test_grasp()

	print "TRANSITION SCORE ", test_transistion()
