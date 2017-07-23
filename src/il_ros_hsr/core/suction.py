#!/usr/bin/python
# -*- coding: utf-8 -*-

import hsrb_interface
import rospy
import sys
import math
import tf
import tf2_ros
import tf2_geometry_msgs
import IPython
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped

from tmc_suction.msg import (
    SuctionControlAction,
    SuctionControlGoal
)

import actionlib

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo, JointState

from il_ros_hsr.core.gripper import VGripper
from image_geometry import PinholeCameraModel as PCM

__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0

class Suction(VGripper):

   

    def start(self):

        rospy.loginfo('Suction will start')
        suction_on_goal = SuctionControlGoal()
        suction_on_goal.timeout = __SUCTION_TIMEOUT__
        suction_on_goal.suction_on.data = True
        self.suction_control_client.send_goal_and_wait(suction_on_goal)




    def stop(self):

        rospy.loginfo('Suction will stop')
        suction_off_goal = SuctionControlGoal()
        suction_off_goal.suction_on.data = False
        self.suction_control_client.send_goal_and_wait(suction_off_goal)

    
    def execute_grasp(self,cards,whole_body):

        
        whole_body.end_effector_frame = 'hand_palm_link'
        nothing = True
        
        #self.whole_body.move_to_neutral()
        whole_body.move_end_effector_pose(geometry.pose(ek = -3.14),cards[0])
        
        self.start()

        whole_body.move_to_joint_positions({'arm_lift_joint':0.23})

       
        #self.whole_body.move_end_effector_pose(geometry.pose(z=-0.9),'hand_l_finger_vacuum_frame')

        self.stop()







if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()