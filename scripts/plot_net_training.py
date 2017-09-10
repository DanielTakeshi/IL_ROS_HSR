import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython
from numpy.random import normal
import time
#import listener
import thread
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
import rospy
import glob
import os
import cPickle as pickle
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')


import il_ros_hsr.p_pi.bed_making.config_bed as cfg



from detectors.grasp_detector import GDetector
from detectors.tran_detector import SDetector

rollouts = []



def plot_grasp_data():

	stats_file = glob.glob(os.path.join(cfg.GRASP_STAT_DIR  , '*CS_*'))

	for stat_file in stats_file:
		data = pickle.load(open(stat_file))

		test_data = data['test']

		train_data = data['train']

		print stat_file
		print train_data
	
		plt.plot(test_data,'--',label=stat_file[-11:-7])
		plt.plot(train_data,label=stat_file[-11:-7])

	plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

	plt.show()



def plot_tran_data():
	stats_file = glob.glob(os.path.join(cfg.TRAN_STATS_DIR  , '*_*'))

	for stat_file in stats_file:
		data = pickle.load(open(stat_file))

		test_data = data['test']

		train_data = data['train']

		plt.plot(train_data)


	plt.show()





if __name__ == "__main__":

	plot_grasp_data()
