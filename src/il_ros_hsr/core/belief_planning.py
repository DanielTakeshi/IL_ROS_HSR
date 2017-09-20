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


__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0

class Belief_Planning(object):

    def __init__(self,base,cam,options):
        #topic_name = '/hsrb/head_rgbd_sensor/depth_registered/image_raw'
        suction_action = '/hsrb/suction_control'

        self.suction_control_client = actionlib.SimpleActionClient(
            suction_action, SuctionControlAction)
     
       
        not_read = True
        while not_read:

            try:
                cam_info = cam.read_info_data()
                if(not cam_info == None):
                    not_read = False
            except:
                rospy.logerr('info not recieved')
       

        self.pcm = PCM()
        self.pcm.fromCameraInfo(cam_info)
        self.br = tf.TransformBroadcaster()
        self.gp = graspPlanner

        self.options = options
        self.base = base

    def broadcast_pose(self,pose):
        #while True: 
        
        count = 0

            
        num_pose = pose[1]
        label = pose[0]

        

        td_points = self.pcm.projectPixelTo3dRay((num_pose[0],num_pose[1]))
        pose = np.array([td_points[0],td_points[1],0.001*num_pose[2]])
        

        self.br.sendTransform((td_points[0], td_points[1], pose[2]),
                tf.transformations.quaternion_from_euler(ai=-2.355,aj=-3.14,ak=0.0),
                rospy.Time.now(),
                'belief',
                'head_rgbd_sensor_rgb_frame')
        count += 1


    def go_to_confident_view(self,boxes,c_img,d_img):
        
        p_list = []
        mc_result = results[0]
        for result in results:
            if(result['prob'] > mc_result['prob']):
                mc_result = result

            print result



        x = int(mc_result['box'][0])
        y = int(mc_result['box'][1])
        w = int(mc_result['box'][2]/ 2.0)
        h = int(mc_result['box'][3]/ 2.0)

        #Crop D+img
        
        
        #Crop D+img
        d_img_c = d_img[y-h:y+h,x-w:x+w]

        depth = self.gp.find_max_depth(d_img_c)
        pose = [mc_result['class'],self.convert_crop([x,y,depth])]

        self.broadcast_poses(poses)

    def go_to_pose(self,results,c_img,d_img):
        

        whole_body.end_effector_frame = 'hand_l_finger_vacuum_frame'
        nothing = True
        
        #self.whole_body.move_to_neutral()
        #whole_body.linear_weight = 99.0
        transforms = self.tl.getFrameStrings()
    
        cards = []
        
        for transform in transforms:
            if 'belief' in transform:
                f_p = self.tl.lookupTransform('head_rgbd_sensor_rgb_frame',transform, rospy.Time(0))
                cards.append(transform)
        
                return True, cards

        whole_body.move_end_effector_pose(geometry.pose(),cards[0])

        self.base.move(geometry.pose(),100.0,ref_frame_id = 'belief')









if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()