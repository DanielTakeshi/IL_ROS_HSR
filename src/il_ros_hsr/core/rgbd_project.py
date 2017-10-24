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
import time

from il_ros_hsr.core.sensors import  RGBD

import cv2

import numpy as np



from image_geometry import PinholeCameraModel as PCM
import thread

from  numpy.random import multivariate_normal as mvn
import il_ros_hsr.p_pi.bed_making.config_bed as cfg


__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0


class RGBD_Project(object):

    def __init__(self):        
        self.cam = RGBD()

        not_read = True
        while not_read:

            try:
                cam_info = self.cam.read_info_data()
                if(not cam_info == None):
                    not_read = False
            except:
                rospy.logerr('info not recieved')
       

        self.pcm = PCM()
  
        self.pcm.fromCameraInfo(cam_info)
      

    
    
    def deproject_pose(self,pose):
        """"
        pose = (u_x,u_y,z)

        u_x,u_y correspond to pixel value in image
        x corresponds to depth
        """
        
            
        td_points = self.pcm.projectPixelTo3dRay((pose[0],pose[1]))

        norm_pose = np.array(td_points)
        norm_pose = norm_pose/norm_pose[2]
        norm_pose = norm_pose*(cfg.MM_TO_M*pose[2])
        

        return norm_pose
        
       
        


   
       

  




if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()