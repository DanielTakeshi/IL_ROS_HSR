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



class Collect_Demos():

	def __init__(self):
		self.robot = hsrb_interface.Robot()

		self.noise = 0.1

		self.omni_base = self.robot.get('omni_base')
		self.whole_body = self.robot.get('whole_body')
		self.gripper = self.robot.get('gripper')
		self.tl = TransformListener()  

		self.start_recording = False
		self.stop_recording = False

		self.com = COM()

		#self.com.go_to_initial_state(self.whole_body,self.gripper)

		#self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.3})

		self.cam = RGBD()

		self.joystick = JoyStick_X(self.com)
		self.torque = Gripper_Torque()
		self.joints = Joint_Positions()


	
	

	def proess_state(self):

		img_rgb = self.cam.read_color_data()
		img_depth = self.cam.read_depth_data()

		cur_action,time_pub = self.joystick.apply_control()

		state = self.com.format_data(img_rgb,img_depth)

		

		#cv2.imwrite('frame_'+str(self.count)+'.png',state[0])
		#Save all data
		data = {}
		data['action'] = cur_action
		data['color_img'] = state[0]
		data['depth_img'] = state[1]
		data['noisey_twist'] = cur_action
		data['gripper_torque'] = self.torque.read_data()
		data['joint_positions'] = self.joints.read_data()
		data['action_time'] = time_pub
		data['image_time'] = self.cam.color_time_stamped

		print "ACTION AT COUNT ",self.count
		print cur_action
		self.count += 1
		self.trajectory.append(data)
		# if(LA.norm(cur_action) > 1e-3):
		# 	print "GOT ACCEPTED"
		# 	self.trajectory.append(data)
		


	def run(self):

		
		cur_action = self.joystick.apply_control()
		cur_recording = self.joystick.get_record_actions()
			
		

		if(cur_recording[0] < -0.1):
			print "BEGIN DATA COLLECTION"
			self.start_recording = True
		count = 0
		
		if(self.start_recording):
			self.count = 0
			self.trajectory = []
			while not self.stop_recording:
			#while count < 20:

				if(self.cam.is_updated):
					self.proess_state()
					self.cam.is_updated = False


				cur_recording = self.joystick.get_record_actions()
				if(cur_recording[1] < -0.1):
					print "END DATA COLLECTION"
					self.stop_recording = True

				count += 1

			q = input('SAVE DATA: y or n: ')

			if(q == 'y'):
				self.com.save_recording(self.trajectory)

			self.start_recording = False
			self.stop_recording = False



			



if __name__=='__main__':
	cd = Collect_Demos()
	time.sleep(5)
	while True: 
		cd.run()