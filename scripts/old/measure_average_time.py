from hsrb_interface import geometry
import hsrb_interface
import IPython
import time
#import listener
import thread



import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import tf
import rospy

from il_ros_hsr.core.grasp_planner import GraspPlanner

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys, os
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
# from yolo.detector import Detector
from online_labeler import QueryLabeler

import il_ros_hsr.p_pi.bed_making.config_bed as cfg
import cPickle as pickle

from il_ros_hsr.core.rgbd_to_map import RGBD2Map

from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction

dp = DrawPrediction()
#latest, 46-49 from rollout_dart
prev_time = 0.0
dur_times = []
for rnum in range(30,40):
	# path = cfg.STAT_PATH+'stat_' + str(rnum) + '/rollout.p'
	path = cfg.ROLLOUT_PATH+'rollout_' + str(rnum) + '/rollout.p'

	stat = os.stat(path)

	print "PATH ",path
	if prev_time > 0.0:
		dur_time = stat.st_ctime - prev_time
		print "DURATION ",dur_time
		dur_times.append(dur_time)

	prev_time = stat.st_ctime


print "AVERAGE TIME ", np.mean(dur_times)/60.0
	


