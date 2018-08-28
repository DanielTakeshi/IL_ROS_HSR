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
from yolo.detector import Detector
from online_labeler import QueryLabeler
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler

from il_ros_hsr.p_pi.bed_making.check_success import Success_Check
from il_ros_hsr.p_pi.bed_making.self_supervised import Self_Supervised
import il_ros_hsr.p_pi.bed_making.config_bed as cfg


from il_ros_hsr.core.rgbd_to_map import RGBD2Map
class Reward_Measure():

	def __init__(self):
		'''
		Initialization class for a Policy

		Parameters
		----------
		yumi : An instianted yumi robot 
		com : The common class for the robot
		cam : An open bincam class

		debug : bool 

		A bool to indicate whether or not to display a training set point for 
		debuging. 

		'''
		self.cam = RGBD()
		self.com = COM()

		not_read = True
		while not_read:

			try:
				cam_info = self.cam.read_info_data()
				if(not cam_info == None):
					not_read = False
			except:
				rospy.logerr('info not recieved')


		#self.side = 'BOTTOM'
		self.cam_info = cam_info
		self.cam = RGBD()
		self.com = COM()

	def capture_data(self):

		data = []
		while True:
			c_img = self.cam.read_color_data()
			d_img = self.cam.read_depth_data()
			if(not c_img == None and not d_img == None):
				datum = {}
				cv2.imshow('debug',c_img)
				cv2.waitKey(30)
				datum['c_img'] = c_img
				datum['d_img'] = d_img
				data.append(datum)
				IPython.embed()

	def position_head(self,whole_body):
		whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})




if __name__ == "__main__":
   
	robot = hsrb_interface.Robot()
	omni_base = robot.get('omni_base')
	whole_body = robot.get('whole_body')

	com = COM()

	# com.go_to_initial_state(whole_body)

	# tt = TableTop()
	# tt.find_table(robot)

	# tt.move_to_pose(omni_base,'right_down')
	# tt.move_to_pose(omni_base,'right_mid')

	rm = Reward_Measure()
	rm.position_head(whole_body)
	rm.capture_data()

