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
from il_ros_hsr.p_pi.safe_corl.features import Features
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"



class FindObject():

	def __init__(self,features,user_name = None):
		
		self.com = COM()
		self.robot = hsrb_interface.Robot()

		self.noise = 0.1
		self.features = features#self.com.binary_image
		self.count = 0
	
		if(not user_name == None):
			self.com.Options.setup(self.com.Options.root_dir,user_name)

		

		self.omni_base = self.robot.get('omni_base')
		self.whole_body = self.robot.get('whole_body')
		self.gripper = self.robot.get('gripper')

		self.com.go_to_initial_state(self.whole_body,self.gripper)

		self.joystick = JoyStick_X(self.com)

		self.policy = Policy(self.com,self.features)

	def run(self):

		if(self.policy.cam.is_updated):
			self.com.load_net()
			self.policy.rollout()
			self.com.clean_up()

			self.policy.cam.is_updated = False
			self.count += 1
		

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


	def go_to_initial_state(self):
		self.com.go_to_initial_state(self.whole_body,self.gripper)


	def check_success(self):

		self.com.clean_up()

		self.b_d = Bottle_Detect()

		img_rgb = self.policy.cam.read_color_data()

		success = self.b_d.detect_bottle(img_rgb)

		self.b_d.clean_up()

		print "BOTTLE FOUND ",success

		return success

	def save_data(self,img_rgb,img_detect):

		state = self.com.format_data(img_rgb,None)

		state['found_object'] = img_detect
		state['object_poses'] = 
	def execute_grasp(self):

		self.gripper.grasp(0.01)
		self.whole_body.end_effector_frame = 'hand_palm_link'
		nothing = True
		while nothing: 
			try:
				self.whole_body.move_end_effector_pose(geometry.pose(y=0.15,z=-0.11, ek=-1.57),'ar_marker/9')
				nothing = False
			except:
				rospy.logerr('mustard bottle found')

		try:
			self.gripper.grasp(-0.5)
		except:
			rospy.logerr('grasp error')

		self.whole_body.move_end_effector_pose(geometry.pose(z=-0.9),'hand_palm_link')

		self.gripper.grasp(0.01)
		self.gripper.grasp(-0.01)





if __name__=='__main__':
	
	
	

	features = Features()
	username = 'corl_anne_n1/'
	fo = FindObject(features.vgg_features,user_name = username)
	

	
	time.sleep(5)
	count = 0
	T = 20
	grasp_on = False

	while True:
		success = False
		e_stop = False
		fo.count = 0
		fo.check_initial_state()

		

		while fo.count < T and not success and not e_stop: 
			fo.run()
			success = fo.check_success()

			
			cur_recording = fo.joystick.get_record_actions_passive()
			if(cur_recording[0] < -0.1):
				print "E-STOPPED"
				e_stop = True

			
			print "CURRENT COUNT ",count
			if success and grasp_on:
				fo.execute_grasp()
				fo.go_to_initial_state()


	


	