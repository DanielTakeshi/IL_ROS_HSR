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



class Collect_Demos():

	def __init__(self,user_name = None,inject_noise = False,noise_scale = 1.0):
		self.robot = hsrb_interface.Robot()

		self.noise = 0.1

		self.omni_base = self.robot.get('omni_base')
		self.whole_body = self.robot.get('whole_body')
		self.gripper = self.robot.get('gripper')
		self.tl = TransformListener()  
		self.b_d = Bottle_Detect()

		self.start_recording = False
		self.stop_recording = False

		self.com = COM()

		if(not user_name == None):
			self.com.Options.setup(self.com.Options.root_dir,user_name)

		self.com.go_to_initial_state(self.whole_body,self.gripper)

		#self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.3})

		self.cam = RGBD()

		self.joystick = JoyStick_X(self.com,inject_noise = inject_noise,noise_scale = noise_scale)
		self.torque = Gripper_Torque()
		self.joints = Joint_Positions()


	
	

	def proess_state(self):

		img_rgb = self.cam.read_color_data()
		img_depth = self.cam.read_depth_data()

		cur_action,noise_action,time_pub = self.joystick.apply_control()

		state = self.com.format_data(img_rgb,img_depth)

		

		#cv2.imwrite('frame_'+str(self.count)+'.png',state[0])
		#Save all data
		data = {}
		data['action'] = cur_action
		data['color_img'] = state[0]
		data['depth_img'] = state[1]
		data['noise_action'] = noise_action

		pose = self.whole_body.get_end_effector_pose().pos
		pose = np.array([pose.x,pose.y,pose.z])
		data['robot_pose'] = pose
		# data['action_time'] = time_pub
		# data['image_time'] = self.cam.color_time_stamped

		print "ACTION AT COUNT ",self.count
		print cur_action
		self.count += 1
		self.trajectory.append(data)
		# if(LA.norm(cur_action) > 1e-3):
		# 	print "GOT ACCEPTED"
		# 	self.trajectory.append(data)


	def check_success(self):

		img_rgb = self.cam.read_color_data()

		success = self.b_d.detect_bottle(img_rgb)

		print "BOTTLE FOUND ",success

		return success
		


	def run(self):

		
		self.joystick.apply_control()
		cur_recording = self.joystick.get_record_actions_passive()
		
		

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

			self.check_success()
			q = input('SAVE DATA: y or n: ')

			if(q == 'y'):
				self.com.save_recording(self.trajectory)

			self.start_recording = False
			self.stop_recording = False



			



if __name__=='__main__':
	user_name = 'corl_matt_n1/'
	noise_scale = 3.0
	inject_noise = True

	cd = Collect_Demos(user_name,inject_noise=inject_noise,noise_scale = noise_scale)

	time.sleep(5)
	while True: 
		cd.run()