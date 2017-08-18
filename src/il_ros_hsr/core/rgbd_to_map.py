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

import actionlib
import time
import thread
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np



__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0


class RGBD2Map(object):

    def __init__(self):
        #topic_name = '/hsrb/head_rgbd_sensor/depth_registered/image_raw'
        

        self.br = tf.TransformBroadcaster()

        thread.start_new_thread(self.broadcast,())
       

    




    def broadcast(self):

        trans = np.array([0.009, 0.008, -0.024])    
        #0.084, -0.003, -0.007
        quat = tf.transformations.quaternion_from_euler(ai=0.084,aj=-0.003,ak=-0.007)
        #IPython.embed()

        while True:
            self.br.sendTransform(trans,
                quat,
                rospy.Time.now(),
                'rgbd_sensor_rgb_frame_map',
                'head_rgbd_sensor_link')



if __name__=='__main__':
    robot =  hsrb_interface.Robot()

    rgbd_map = RGBD2Map()
    