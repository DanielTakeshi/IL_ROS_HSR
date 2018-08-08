from hsrb_interface import geometry
import hsrb_interface
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
import geometry_msgs
import controller_manager_msgs.srv
import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython
from numpy.random import normal
import time
#import listener
import thread

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import tf
import rospy

from il_ros_hsr.core.grasp_planner import GraspPlanner

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
# from yolo.detector import Detector
from online_labeler import QueryLabeler
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler

from il_ros_hsr.p_pi.bed_making.check_success import Success_Check
from il_ros_hsr.p_pi.bed_making.self_supervised import Self_Supervised
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
import cPickle as pickle
import os
import glob
from il_ros_hsr.core.rgbd_to_map import RGBD2Map

from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction



dp = DrawPrediction()
#latest, 46-49 from rollout_dart
sm = 0
com = COM()
wl = Python_Labeler()

# Meant to be run _after_ the `collect_data_bed_fast.py` script which saves these images.
for rnum in range(60,65):
    # path = cfg.STAT_PATH+'stat_' + str(rnum) + '/rollout.p'

    b_grasps = glob.glob(cfg.FAST_PATH+'b_grasp/*_'+str(rnum)+'_*')
    b_success = glob.glob(cfg.FAST_PATH+'b_success/*_'+str(rnum)+'_*')

    t_grasps = glob.glob(cfg.FAST_PATH+'t_grasp/*_'+str(rnum)+'_*')
    t_success = glob.glob(cfg.FAST_PATH+'t_success/*_'+str(rnum)+'_*')

    rollout_data = []
    print "------GRASP-----------"
    for grasp in b_grasps:
        datum = {}
        img = cv2.imread(grasp)
        data = wl.label_image(img)
        label = data['objects'][0]['box']
        pose = [(label[2]-label[0])/2.0+label[0],(label[3]-label[1])/2.0+label[1]]

        datum['c_img'] = img
        datum['class'] = data['objects'][0]['class']
        datum['side'] = 'BOTTOM'
        datum['type'] = 'grasp'
        datum['pose'] = pose
        rollout_data.append(datum)

    for grasp in t_grasps:
        datum = {}
        img = cv2.imread(grasp)
        data = wl.label_image(img)
        label = data['objects'][0]['box']
        pose = [(label[2]-label[0])/2.0+label[0],(label[3]-label[1])/2.0+label[1]]

        datum['c_img'] = img
        datum['class'] = data['objects'][0]['class']
        datum['side'] = 'TOP'
        datum['type'] = 'grasp'
        datum['pose'] = pose
        rollout_data.append(datum)

    print "------SUCCESSS-----------"
    for success in b_success:
        datum = {}
        img = cv2.imread(success)
        data = wl.label_image(img)
        label = data['objects'][0]['box']
        pose = [(label[2]-label[0])/2.0+label[0],(label[3]-label[1])/2.0+label[1]]

        datum['c_img'] = img
        datum['class'] = data['objects'][0]['class']
        datum['side'] = 'BOTTOM'
        datum['type'] = 'success'
        datum['pose'] = pose
        rollout_data.append(datum)

    for success in t_success:
        datum = {}
        img = cv2.imread(success)
        data = wl.label_image(img)
        label = data['objects'][0]['box']
        pose = [(label[2]-label[0])/2.0+label[0],(label[3]-label[1])/2.0+label[1]]

        datum['c_img'] = img
        datum['class'] = data['objects'][0]['class']
        datum['side'] = 'TOP'
        datum['type'] = 'success'
        datum['pose'] = pose
        rollout_data.append(datum)

    com.save_rollout(rollout_data)
