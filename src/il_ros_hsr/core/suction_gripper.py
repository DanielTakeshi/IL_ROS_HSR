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
import il_ros_hsr.p_pi.tpc.config_tpc as cfg
import thread

from  numpy.random import multivariate_normal as mvn


__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0


class Suction_Gripper(object):

    def __init__(self,graspPlanner,cam,options,suction):
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
        self.options = options
        self.pcm.fromCameraInfo(cam_info)
        self.br = tf.TransformBroadcaster()
        self.tl = tf.TransformListener()
        self.gp = graspPlanner
        self.suction = suction
        self.com = COM()
        self.count = 0


    def compute_trans_to_map(self,norm_pose,rot):
        time.sleep(0.5)
        pose = self.tl.lookupTransform('map','rgbd_sensor_rgb_frame_map', rospy.Time(0))
       
        M = tf.transformations.quaternion_matrix(pose[1])
        M_t = tf.transformations.translation_matrix(pose[0])
        M[:,3] = M_t[:,3]


        M_g = tf.transformations.quaternion_matrix(rot)
        M_g_t = tf.transformations.translation_matrix(norm_pose)
        M_g[:,3] = M_g_t[:,3] 

        M_T = np.matmul(M,M_g)

        trans = tf.transformations.translation_from_matrix(M_T)

        quat = tf.transformations.quaternion_from_matrix(M_T)

        return trans,quat

    def loop_broadcast(self,norm_pose,base_rot,rot_z):
        norm_pose,rot = self.compute_trans_to_map(norm_pose,base_rot)
        print "NORM POSE ",norm_pose
        count = np.copy(self.count)
        while True:
            self.br.sendTransform((norm_pose[0], norm_pose[1], norm_pose[2]),
                    #tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
                    rot,
                    rospy.Time.now(),
                    'suction_i_'+str(count),
                    #'head_rgbd_sensor_link')
                    'map')

            
            self.br.sendTransform((0.0, 0.0, -cfg.GRIPPER_HEIGHT),
                    tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=rot_z),
                    rospy.Time.now(),
                    'suction'+str(count),
                    #'head_rgbd_sensor_link')
                    'suction_i_'+str(count))
    
    def broadcast_poses(self,position,rot):
        #while True: 
        
        count = 0
        
        td_points = self.pcm.projectPixelTo3dRay((position[0],position[1]))
        print "DE PROJECTED POINTS ",td_points
        norm_pose = np.array(td_points)
        norm_pose = norm_pose/norm_pose[2]
        norm_pose = norm_pose*(cfg.MM_TO_M*position[2])
        print "NORMALIZED POINTS ",norm_pose
        
        #pose = np.array([td_points[0],td_points[1],0.001*num_pose[2]])
        a = tf.transformations.quaternion_from_euler(ai=-2.355,aj=-3.14,ak=0.0)
        b = tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=1.57)

        base_rot = tf.transformations.quaternion_multiply(a,b)
    
        thread.start_new_thread(self.loop_broadcast,(norm_pose,base_rot,rot))

        time.sleep(0.8)

    def convert_crop(self,pose):

        pose[0] = self.options.OFFSET_Y + pose[0]
        pose[1] = self.options.OFFSET_X + pose[1]

        return pose

    def plot_on_true(self,pose,true_img):

        #pose = self.convert_crop(pose)

        # dp = DrawPrediction()

        # image = dp.draw_prediction(np.copy(true_img),pose)

        # cv2.imshow('label_given',image)

        # cv2.waitKey(30)

       pass

    def get_grasp_pose(self,x,y,z,rot,c_img=None):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''

        poses = []
        #IPython.embed()
        
        p_list = []

        pose = [x,y,z]
        self.plot_on_true([x,y],c_img)
        rot = 0
        self.broadcast_poses(pose,rot)

        grasp_name = 'suction'+str(self.count)
        self.count += 1

        return grasp_name



    def start(self):
        try:
            self.suction.command(2)
        except Exception as e:
            IPython.embed()
            rospy.logerr('suction open error')


    def stop(self):
        try:
            self.suction.command(0)
        except:
            rospy.logerr('suction close error')


if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()
