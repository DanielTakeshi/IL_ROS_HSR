import os
import Pyro4
import time
import cPickle as pickle
import IPython
import cv2

from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler
import hsrb_interface

from il_ros_hsr.core.sensors import  RGBD

#robot interface
GLOBAL_PATH = "/home/autolab/Workspaces/michael_working/IL_ROS_HSR/"
CANVAS_DIM = 420.0

class Success_Check:

	def __init__(self,whole_body,tt,cam,base):

		self.cam = cam
		self.whole_body = whole_body
		self.tt = tt
		self.omni_base = base

	def check_bottom_success(self,wl):
		self.whole_body.move_to_go()
		self.tt.move_to_pose(self.omni_base,'lower_mid')
		self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})

		return self.check_success(wl)

	def check_top_success(self,wl):
		self.whole_body.move_to_go()
		self.tt.move_to_pose(self.omni_base,'top_mid')
		self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})

		return self.check_success(wl)



	def check_success(self,wl):

		img = self.cam.read_color_data()
		data = wl.label_image(img)

		for result in data['objects']:

			if(result['class'] == 0):
				return True
			else:
				return False






if __name__ == "__main__":

	robot = hsrb_interface.Robot()

	omni_base = robot.get('omni_base')
	whole_body = robot.get('whole_body')

	cam = RGBD()

	wl = Python_Labeler(cam)



	sc = Success_Check(whole_body,None,cam,omni_base)

	print "RESULT ", sc.check_success(wl)
		
