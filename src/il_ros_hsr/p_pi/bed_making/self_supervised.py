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

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM

from il_ros_hsr.p_pi.bed_making.tensioner import Tensioner

from il_ros_hsr.core.sensors import Gripper_Torque
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
import thread

from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction

import numpy.linalg as LA

__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0


class Self_Supervised(object):

    def __init__(self,cam):
        #topic_name = '/hsrb/head_rgbd_sensor/depth_registered/image_raw'
        


        not_read = True
        while not_read:

            try:
                cam_info = cam.read_info_data()
                if(not cam_info == None):
                    not_read = False
            except:
                rospy.logerr('info not recieved')
       

        self.pcm = PCM()

        self.cam = cam
    
        self.pcm.fromCameraInfo(cam_info)
        self.br = tf.TransformBroadcaster()
        self.tl = tf.TransformListener()





    def debug_images(self,pose):

        c_img = self.cam.read_color_data()
        dp = DrawPrediction()

        image = dp.draw_prediction(np.copy(c_img),pose)

        cv2.imshow('debug',image)
        cv2.waitKey(30)
        #IPython.embed()


    def get_current_head_position(self):

        pose = self.tl.lookupTransform('rgbd_sensor_rgb_frame_map','map', rospy.Time(0))

        M = tf.transformations.quaternion_matrix(pose[1])
        M_t = tf.transformations.translation_matrix(pose[0])
        M[:,3] = M_t[:,3]

        return M


    def compute_transform(self,M_g):

        M_curr = self.get_current_head_position()

        M_d = np.matmul(M_curr,LA.inv(self.M_base))

        M_T = np.matmul(M_d,M_g)


        trans = M_T[0:3,3]

        
        return trans


    def search_head(self,whole_body,o_p,o_t):

        whole_body.move_to_joint_positions({'head_pan_joint': self.c_p+o_p})
        whole_body.move_to_joint_positions({'head_tilt_joint':self.c_t+o_t})

        time.sleep(cfg.SS_TIME)




    def learn(self,whole_body,grasp_count,cam=None):

        self.d_points = []
        self.M_base = self.get_current_head_position() 

        self.c_p = whole_body.joint_state.position[9]
        self.c_t = whole_body.joint_state.position[10]
        
        change = np.linspace(-0.05,0.05,num =cfg.NUM_SS_DATA)

        for c_t in change:
            for c_p in change:

                self.search_head(whole_body,c_p,c_t)


                M_g = self.look_up_transform(grasp_count)

                transform = M_g[0:3,3]
                
                pose = self.pcm.project3dToPixel(transform)

                self.add_data_point(pose)
                self.debug_images(pose)

        return self.d_points



    def add_data_point(self,pose):

        d_point = {}

        d_point['c_img'] = self.cam.read_color_data()
        d_point['d_img'] = self.cam.read_depth_data()
        d_point['pose'] = pose

        self.d_points.append(d_point)

    def look_up_transform(self,count):

        transforms = self.tl.getFrameStrings()

        for transform in transforms:
            current_grasp = 'bed_i_'+str(count)
            if current_grasp in transform:
                print 'got here'
                pose = self.tl.lookupTransform('rgbd_sensor_rgb_frame_map',transform, rospy.Time(0))


        M = tf.transformations.quaternion_matrix(pose[1])
        M_t = tf.transformations.translation_matrix(pose[0])
        M[:,3] = M_t[:,3]

        return M




        






if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()
