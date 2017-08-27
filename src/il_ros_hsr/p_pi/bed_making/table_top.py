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


# IPython.embed()
# from tmc_navigation_msgs import OccupancyGridUnit

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
from tf import TransformBroadcaster
import tf
import rospy

from hsrb_interface.collision_world import CollisionWorld

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
from yolo.detector import Detector
from image_geometry import PinholeCameraModel as PCM
import thread

import numpy.linalg as LA
TABLE_HEIGHT = 0.61
TABLE_WIDTH = 0.67

OFFSET = 0.5
OFFSET_T = 0.34
TABLE_LENGTH = 0.91

class TableTop():

	def __init__(self):


		

		self.tl = TransformListener()
		self.br = TransformBroadcaster()
		
		
		#table = cw.add_box(z = TABLE_WIDTH, x= TABLE_LENGTH, y = TABLE_HEIGHT, 
		#	pose = geometry.pose(x = TABLE_LENGTH/2.0, y = TABLE_HEIGHT/2.0, z = TABLE_WIDTH /2.0), frame_id = 'ar_marker/14' )

	

	def broadcast_pose(self,pose,label):

		while True:
			 self.br.sendTransform(pose['trans'],
                    pose['quat'],
                    rospy.Time.now(),
                    label,
                    'map')
	
	def move_to_pose(self,base,label):

		base.move(geometry.pose(),500.0,ref_frame_id = label)

	def cal_transform(self,offsets,rot = None):
		L_t_trans = tf.transformations.translation_matrix(offsets)

		M_t_L = np.matmul(self.M_t_A,L_t_trans)

		if not rot == None:
			q_rot = tf.transformations.quaternion_from_euler(ai=rot[0],aj=rot[1],ak=rot[2])
			L_t_rot = tf.transformations.quaternion_matrix(q_rot)
			L_t_rot[:,3] = L_t_trans[:,3]

			M_t_L = np.matmul(self.M_t_A,L_t_rot)



		

		trans = tf.transformations.translation_from_matrix(M_t_L)

 		quat = tf.transformations.quaternion_from_matrix(M_t_L)


 		return trans, quat





	def calculat_ar_in_map(self,obj):

		ar_pose = obj.get_pose(ref_frame_id = 'ar_marker/11')
		marker_pose = obj.get_pose(ref_frame_id = 'map')

		
		M_t_O = tf.transformations.quaternion_matrix(marker_pose.ori)
		M_t_trans = tf.transformations.translation_matrix(marker_pose.pos)
		M_t_O[:,3] = M_t_trans[:,3]

		A_t_O = tf.transformations.quaternion_matrix(ar_pose.ori)
		A_t_trans = tf.transformations.translation_matrix(ar_pose.pos)
		A_t_O[:,3] = A_t_trans[:,3]

		self.M_t_A = np.matmul(M_t_O,LA.inv(A_t_O))

		trans = tf.transformations.translation_from_matrix(self.M_t_A)

 		quat = tf.transformations.quaternion_from_matrix(self.M_t_A)

 		print "TRANS ",trans
 		print "ROTATION ",quat

 		return trans,quat



 	def make_new_pose(self,offsets,label,rot = None):

 		t,q = self.cal_transform(offsets,rot = rot)
		# top_corner_trans[1] = top_corner_trans[1] + (2*OFFSET+TABLE_WIDTH)
		# top_corner_trans[0] = top_corner_trans[0] + (OFFSET+TABLE_LENGTH/2.0)

		pose = {}
		pose['trans'] = t
		pose['quat'] = q


		thread.start_new_thread(self.broadcast_pose,(pose,label))


	def find_table(self,robot):

		objts = robot.get('marker')

		time.sleep(5)
		#IPython.embed()
		sd = objts.get_objects()
		
		trans, quat = self.calculat_ar_in_map(sd[0])

		
		
		#Compute TOP MID
		offsets = np.array([0.0, (2*OFFSET+TABLE_WIDTH), 0.0])
		rot = np.array([0.0,0.0,-1.57])
		self.make_new_pose(offsets,'top_mid',rot=rot)


		#Compute TOP CORNER
		offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0), (2*OFFSET+TABLE_WIDTH), 0.0])
		self.make_new_pose(offsets,'right_up')


		#Compute RIGHT CORNER
		offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0), 0.0, 0.0])
		self.make_new_pose(offsets,'right_down')



		#Compute RIGHT MID
		offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0),(OFFSET+TABLE_WIDTH/2.0),0.0])
		self.make_new_pose(offsets,'right_mid')
	

		#Compute LOWER MID 
		offsets = np.array([0.0,-OFFSET-0.07,0.0])
		rot = np.array([0.0,0.0,-3.14])
		self.make_new_pose(offsets,'lower_mid',rot=rot)

		#Compute HEAD DOWN
		offsets = np.array([-(TABLE_LENGTH/2.0), OFFSET_T+0.04, -TABLE_HEIGHT])
		rot = np.array([0.0,0.0,1.57])
		self.make_new_pose(offsets,'head_down',rot = rot)

		#Compute HEAD UP
		offsets = np.array([-(TABLE_LENGTH/2.0), (OFFSET_T+TABLE_WIDTH+0.02), -TABLE_HEIGHT-0.02])
		rot = np.array([0.0,0.0,-1.57])
		self.make_new_pose(offsets,'head_up',rot = rot)

		#Compute LOWER MID 
		offsets = np.array([0.0,-OFFSET+0.16,0.0])
		rot = np.array([0.0,0.0,1.57])
		self.make_new_pose(offsets,'lower_start',rot=rot)



if __name__ == "__main__":

	robot = hsrb_interface.Robot()

	omni_base = robot.get('omni_base')
	whole_body = robot.get('whole_body')

	com = COM()
	com.go_to_initial_state(whole_body)


	tt = TableTop()
	tt.find_table(robot)


	
	IPython.embed()
	tt.move_to_pose(omni_base,'lower_mid')
	# tt.move_to_pose(omni_base,'right_corner')
	# tt.move_to_pose(omni_base,'right_mid')
	

