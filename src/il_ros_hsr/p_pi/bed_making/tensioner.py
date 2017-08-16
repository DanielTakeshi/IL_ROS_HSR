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
import numpy as np
import numpy.linalg as LA
import rospy

from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM

from il_ros_hsr.core.sensors import Gripper_Torque
import il_ros_hsr.p_pi.bed_making.config_bed as cfg



from tf import TransformListener


class Tensioner(object):

    def __init__(self):
        #topic_name = '/hsrb/head_rgbd_sensor/depth_registered/image_raw'
        
       
        self.torque = Gripper_Torque()

        self.tl = TransformListener()
   



    def get_translation(self,direction):

        pose = self.tl.lookupTransform(direction,'hand_palm_link',rospy.Time(0))

        trans = pose[0]

        return  trans

    def get_rotation(self,direction):
        pose = self.tl.lookupTransform(direction,'hand_palm_link',rospy.Time(0))

        rot = pose[1]

        return  tf.transformations.quaternion_matrix(rot)

    def compute_bed_tension(self,wrench,direction):

        x = wrench.wrench.force.x
        y = wrench.wrench.force.y
        z = wrench.wrench.force.z 

        force = np.array([x,y,z,1.0])

        rot = self.get_rotation(direction)
        
        force_perp = np.dot(rot,force)

        print "CURRENT FORCES ",force
        print "FORCE PERP ",force_perp

        return force_perp[1]

    def force_pull(self,whole_body,direction):

        count = 0
        is_pulling = False
        t_o = self.get_translation(direction)
        delta = 0.0
        while True:
            s = 1.0-delta

            whole_body.move_end_effector_pose(geometry.pose(x = t_o[0]*s, y = t_o[1]*s,z= t_o[2]*s),direction)

            wrench = self.torque.read_data()
            norm = np.abs(self.compute_bed_tension(wrench,direction))

            print "FORCE NORM ", norm
     

            if(norm > cfg.HIGH_FORCE):
                is_pulling = True

            if(is_pulling and norm < cfg.LOW_FORCE):
                break

            if norm > cfg.FORCE_LIMT or count == cfg.MAX_PULLS:
                break

            count += 1
            delta += 0.2

        






if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()