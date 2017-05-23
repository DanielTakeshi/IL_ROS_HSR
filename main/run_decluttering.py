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

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick_X import  JoyStick_X

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import rospy

from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM
from il_ros_hsr.p_pi.safe_corl.bottle_detector import Bottle_Detect
from il_ros_hsr.core.policy import Policy
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"



class FindObject():

	def __init__(self):
		
		self.com = COM(load_net = True)
		self.robot = hsrb_interface.Robot()

		self.noise = 0.1

		

		self.omni_base = self.robot.get('omni_base')
		self.whole_body = self.robot.get('whole_body')
		self.gripper = self.robot.get('gripper')
		#self.com.go_to_initial_state(self.whole_body,self.gripper)

		self.joystick = JoyStick_X(self.com)

		

		self.policy = Policy(self.com)

	def run(self):
		self.policy.rollout()
		

	def check_initial_state(self):

		go = False

		while not go:
			img_rgb = self.policy.cam.read_color_data()
			img_depth = self.policy.cam.read_depth_data()
			
			state = self.com.format_data(img_rgb,img_depth)
			cv2.imshow('initial_state', state[0] )
			cv2.waitKey(30)
			self.joystick.apply_control()
			cur_recording = self.joystick.get_record_actions()
			if(cur_recording[1] < -0.1):
				print "BEGIN ROLLOUT"
				go = True




	def check_success(self):

		self.com.clean_up()

		self.b_d = Bottle_Detect()

		img_rgb = self.policy.cam.read_color_data()

		success = self.b_d.detect_bottle(img_rgb)

		print "BOTTLE FOUND ",success





if __name__=='__main__':
	fo = FindObject()
	time.sleep(5)
	count = 0
	T = 40
	fo.check_initial_state()
	

	while count < T: 
		fo.run()
		count += 1
		print "CURRENT COUNT ",count

	fo.check_success()

	