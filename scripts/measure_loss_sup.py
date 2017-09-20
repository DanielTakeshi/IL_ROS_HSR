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


sdect = SDetector(cfg.TRAN_NET_NAME)
gdect = GDetector(cfg.GRASP_NET_NAME)

def load_held_out_rollouts():

	rollouts_p = glob.glob(os.path.join(cfg.BC_HELD_OUT, '*_*'))


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

				label = np.array(d_point['pose'])

				#label = unscale(label,d_point['c_img'])

				out_pred = np.array(gdect.predict(d_point['c_img']))

				
				
				#out_pred = unscale(out_pred,d_point['c_img'])

				l2_grasp_score.append(np.sum((np.square(label-out_pred))**0.5))

	return np.mean(l2_grasp_score)


def compute_covariance():

	l2_grasp_score = []

	N = len(rollouts)
	cov = np.zeros([2,2])

	num_count = 0.0

	for rollout in rollouts:

		for d_point in rollout:

			if type(d_point) == list:
				continue

			if d_point['type'] == 'grasp':

				label = np.zeros([2,1])

				label[:,0] = np.array(d_point['pose'])



				out_pred = np.zeros([2,1])

				out_pred[:,0] = np.array(gdect.predict(d_point['c_img']))

				cov_rank_one = label - out_pred

				num_count += 1.0
			

				cov += np.matmul(cov_rank_one,cov_rank_one.T)

	return cov*1.0/num_count


def test_transistion():

	tran_score = []

	for rollout in rollouts:

		for d_point in rollout:

			if type(d_point) == list:
					continue

			if d_point['type'] == 'success':


				label = d_point['class']

				out_pred = sdect.predict(d_point['c_img'])

				cv2.imshow('image',d_point['c_img'])
				cv2.waitKey(1000)
				print "LABEL ", d_point['class']
				print "OUT_PRED ",out_pred
				print "------------------"
				
				if (np.argmax(label) == np.argmax(out_pred)):
					tran_score.append(0.0)
				else:
					tran_score.append(1.0)


	return np.mean(tran_score)




if __name__ == "__main__":

	load_held_out_rollouts()

	#print "COVARIANCE ", compute_covariance()

	print "GRASP SCORE ", test_grasp()

	#print "TRANSITION SCORE ", test_transistion()
