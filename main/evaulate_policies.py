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

		#self.com.go_to_initial_state(self.whole_body,self.gripper)

		self.joystick = JoyStick_X(self.com)
		self.tl = TransformListener() 

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
		sticky = True
		while  sticky: 
			self.joystick.apply_control()
			cur_recording = self.joystick.get_record_actions()
			if(cur_recording[1] > -0.1):
				print "BEGIN ROLLOUT"
				sticky  = False


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


	def clear_data(self):
		self.trajectory = []

	def mark_success(self,q):

		state = {}
		if(q == 'y'):
			state['success'] = 1
		else:
			state['success'] = 0

		self.trajectory.append(state)

	def is_goal_objet(self):

		
		try:
		
			full_pose = self.tl.lookupTransform('head_l_stereo_camera_frame','ar_marker/9', rospy.Time(0))
			trans = full_pose[0]

			transforms = self.tl.getFrameStrings()
			poses = []


			

			for transform in transforms:
				if 'bottle' in transform:
					f_p = self.tl.lookupTransform('head_rgbd_sensor_link',transform, rospy.Time(0))
					poses.append(f_p[0])
					print 'augmented pose ',f_p
					print 'ar_marker ', trans


			for pose in poses:
				
				if(LA.norm(pose[2]-0.1-trans[2])< 0.03):
					return True
		except: 
			rospy.logerr('AR MARKER NOT THERE')

		return False


	def check_success(self):

		self.com.clean_up()

		self.b_d = Bottle_Detect(self.policy.cam.read_info_data())

		img_rgb = self.policy.cam.read_color_data()
		img_depth = self.policy.cam.read_depth_data()

		s_obj,img_detect,poses = self.b_d.detect_bottle(img_rgb,img_depth)

		success = self.is_goal_objet()

		self.b_d.clean_up()

		self.process_data(img_rgb,img_detect,poses,success)



		print "BOTTLE FOUND ",success

		return success

	def process_data(self,img_rgb,img_detect,object_poses,success):

		img_rgb_cr,img_d = self.com.format_data(img_rgb,None)

		state = {}
		state['color_img'] = img_rgb_cr
		state['found_object'] = img_detect
		state['object_poses'] = object_poses
		state['was_success'] = success

		self.trajectory.append(state)

	def save_data(self):

		self.com.save_evaluation(self.trajectory)


	def check_marker(self):

		try: 
			A = self.tl.lookupTransform('head_l_stereo_camera_frame','ar_marker/9', rospy.Time(0))
		except: 
			rospy.logerr('trash not found')
			return False

		return True


	def execute_grasp(self):

		self.com.grip_open(self.gripper)
		self.whole_body.end_effector_frame = 'hand_palm_link'
		nothing = True
		
		try:
			self.whole_body.move_end_effector_pose(geometry.pose(y=0.15,z=-0.09, ek=-1.57),'ar_marker/9')
			
		except:
			rospy.logerr('mustard bottle found')

		self.com.grip_squeeze(self.gripper)

		self.whole_body.move_end_effector_pose(geometry.pose(z=-0.9),'hand_palm_link')

		self.com.grip_open(self.gripper)
		self.com.grip_squeeze(self.gripper)





if __name__=='__main__':
	
	
	

	features = Features()
	username = 'corl_chris_n1/'
	fo = FindObject(features.vgg_features,user_name = username)
	

	time.sleep(5)


	count = 0
	T = 100
	grasp_on = True

	while True:
		success = False
		e_stop = False
		fo.count = 0
		fo.check_initial_state()
		# fo.execute_grasp()
		# fo.go_to_initial_state()

		
		fo.clear_data()

		while fo.count < T  and not e_stop: 
			fo.run()
			#success = fo.check_success()

			
			cur_recording = fo.joystick.get_record_actions_passive()
			if(cur_recording[0] < -0.1):
				print "E-STOPPED"
				e_stop = True

			
			print "CURRENT COUNT ",count
			if e_stop and grasp_on and fo.check_marker():
				fo.execute_grasp()
				fo.go_to_initial_state()


		q = input('WAS SUCCESFUL: y or n ')

		fo.mark_success(q)

		q = input('SAVE DATA: y or n: ')

		if(q == 'y'):
			fo.save_data()


	


	