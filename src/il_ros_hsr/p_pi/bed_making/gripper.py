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

from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM

from il_ros_hsr.p_pi.bed_making.tensioner import Tensioner

from il_ros_hsr.core.sensors import Gripper_Torque


__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0
GRIPPER_HEIGHT = 90.0
MM_TO_M = 0.001
FORCE_LIMT = 33.0

class Bed_Gripper(object):

    def __init__(self,graspPlanner,cam,options,gripper):
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
        self
        self.pcm.fromCameraInfo(cam_info)
        self.br = tf.TransformBroadcaster()
        self.gp = graspPlanner
        self.gripper = gripper
        self.options = options
        self.com = COM()

        self.tension = Tensioner()

        self.torque = Gripper_Torque()
    
    def broadcast_poses(self,poses):
        #while True: 
        
        count = 0

        
        for pose in poses:
            
            num_pose = pose[1]
            label = pose[0]

            

            td_points = self.pcm.projectPixelTo3dRay((num_pose[0],num_pose[1]))
            print "DE PROJECTED POINTS ",td_points
            norm_pose = np.array(td_points)
            norm_pose = norm_pose/norm_pose[2]
            norm_pose = norm_pose*(MM_TO_M*num_pose[2])
            print "NORMALIZED POINTS ",norm_pose
            
            #pose = np.array([td_points[0],td_points[1],0.001*num_pose[2]])
            a = tf.transformations.quaternion_from_euler(ai=-2.355,aj=-3.14,ak=0.0)
            b = tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=1.57)

            c = tf.transformations.quaternion_multiply(a,b)


            self.br.sendTransform((norm_pose[0], norm_pose[1], norm_pose[2]),
                    #tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
                    c,
                    rospy.Time.now(),
                    'bed_'+str(count),
                    #'head_rgbd_sensor_link')
                    'head_rgbd_sensor_rgb_frame')
            count += 1



    def convert_crop(self,pose):

        pose[0] = self.options.OFFSET_Y + pose[0]
        pose[1] = self.options.OFFSET_X + pose[1]

        return pose

    def plot_on_true(self,pose,true_img):

        #pose = self.convert_crop(pose)

        true_img[pose[1]-5:pose[1]+5,pose[0]-5:pose[0]+5,:] = 0.0
        true_img[pose[1]-5:pose[1]+5,pose[0]-5:pose[0]+5,:] = 255.0
        cv2.imshow('debug_wrap',true_img)

        cv2.waitKey(30)



    def find_pick_region_labeler(self,results,c_img,d_img):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''

        poses = []
        #IPython.embed()
        
        p_list = []
        for result in results['objects']:
            print result

            x_min = float(result['box'][0])
            y_min = float(result['box'][1])
            x_max = float(result['box'][2])
            y_max = float(result['box'][3])

            x = (x_max-x_min)/2.0 + x_min
            y = (y_max - y_min)/2.0 + y_min

            self.plot_on_true([x,y],c_img)
            
            #Crop D+img
            d_img_c = d_img[int(y_min):int(y_max),int(x_min):int(x_max)]

            depth = self.gp.find_max_depth(d_img_c)

            poses.append([result['class'],[x,y,depth]])

        self.broadcast_poses(poses)

    def find_pick_region(self,results,c_img,d_img):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''

        poses = []
        #IPython.embed()
        
        p_list = []
        for result in results:
            print result

            x = int(result['box'][0])
            y = int(result['box'][1])
            w = int(result['box'][2]/ 2.0)
            h = int(result['box'][3]/ 2.0)

            self.plot_on_true([x,y],c_img_true)
            
            
            #Crop D+img
            d_img_c = d_img[y-h:y+h,x-w:x+w]

            depth = self.gp.find_max_depth(d_img_c)
            poses.append([result['class'],self.convert_crop([x,y,depth])])

        self.broadcast_poses(poses)

    def execute_grasp(self,cards,whole_body,direction):

        
        whole_body.end_effector_frame = 'hand_palm_link'
        nothing = True
    
        #self.whole_body.move_to_neutral()
        #whole_body.linear_weight = 99.0
        whole_body.move_end_effector_pose(geometry.pose(z = -GRIPPER_HEIGHT*MM_TO_M),cards[0])

        self.com.grip_squeeze(self.gripper)
        
        self.tension.force_pull(whole_body,direction)
        self.com.grip_open(self.gripper)

        






if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()