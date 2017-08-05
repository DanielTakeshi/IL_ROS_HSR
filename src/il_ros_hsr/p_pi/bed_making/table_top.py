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

from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import Joy

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import tf
import rospy

from hsrb_interface.collision_world import CollisionWorld

from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
from yolo.detector import Detector
from image_geometry import PinholeCameraModel as PCM



class TableTop():

	def __init__(self):


		TABLE_HEIGHT = 0.46
		TABLE_WIDTH = 0.67
		TABLE_LENGTH = 0.91

		self.tl = TransformListener()

		
		
		#table = cw.add_box(z = TABLE_WIDTH, x= TABLE_LENGTH, y = TABLE_HEIGHT, 
		#	pose = geometry.pose(x = TABLE_LENGTH/2.0, y = TABLE_HEIGHT/2.0, z = TABLE_WIDTH /2.0), frame_id = 'ar_marker/14' )

		
	
	def find_table_center(self,robot):

		objts = robot.get('marker')

		time.sleep(5)
		#IPython.embed()
		sd = objts.get_objects()
		
		a = sd[0].get_pose(ref_frame_id = 'ar_marker/11')
		b = sd[0].get_pose(ref_frame_id = 'map')

		trans = TransformStamped()
		trans.header.frame_id = 'map'
		trans.child_frame_id = 'table'
		trans.header.stamp = rospy.Time.now()

		trans.transform.translation.x = 1.0
		trans.transform.translation.y = 1.0
		trans.transform.rotation.w = 1.0
		self.tl.setTransform(trans)
		time.sleep(10)
		a = self.tl.lookupTransform('head_tilt_link','table',rospy.Time(0))
		print a

		IPython.embed()
			# 	rospy.logerr('transform not found')


	def add_table(self,whole_body):

		whole_body.collision_world = cw

		return whole_body

	


if __name__ == "__main__":

	robot = hsrb_interface.Robot()

	omni_base = robot.get('omni_base')
	whole_body = robot.get('whole_body')


	tt = TableTop()
	tt.find_table_center(robot)


	IPython.embed()


