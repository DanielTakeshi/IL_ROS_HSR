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

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.rgbd_to_map import RGBD2Map

from table_top import TableTop
from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction

import numpy.linalg as LA
import numpy as np

from il_ros_hsr.core.Xbox import XboxController

__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0


class InitialSampler(object):

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

        self.xbox = XboxController()





    def debug_images(self,p_1,p_2):

        c_img = self.cam.read_color_data()
        p_1i = (int(p_1[0]),int(p_1[1]))
        p_2i = (int(p_2[0]),int(p_2[1]))
        
        cv2.line(c_img,p_1i,p_2i,(0,0,255),thickness = 10)

        

        cv2.imshow('debug',c_img)
        cv2.waitKey(300)
        #IPython.embed()



    def project_to_rgbd(self,trans):
        M_t = tf.transformations.translation_matrix(trans)

        M_R = self.get_map_to_rgbd()

        M_cam_trans = np.matmul(LA.inv(M_R),M_t)

        

        return M_cam_trans[0:3,3]




    def make_projection(self,t_1,t_2):

        ###GO FROM MAP TO RGBD###

        t_1 = self.project_to_rgbd(t_1)
        t_2 = self.project_to_rgbd(t_2)

        p_1 = self.pcm.project3dToPixel(t_1)
        p_2 = self.pcm.project3dToPixel(t_2)
        
        
        self.debug_images(p_1,p_2)

    def debug_broadcast(self,pose,name):

        while True: 

            self.br.sendTransform((pose[0], pose[1], pose[2]),
                    tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
                    rospy.Time.now(),
                    name,
                    #'head_rgbd_sensor_link')
                    'rgbd_sensor_rgb_frame_map')
            

    def get_map_to_rgbd(self):
        not_found = True
        while not_found:
            try:
                pose = self.tl.lookupTransform('map','rgbd_sensor_rgb_frame_map', rospy.Time(0))
                not_found = False
            except:
                rospy.logerr("waiting for pose")

        M = tf.transformations.quaternion_matrix(pose[1])
        M_t = tf.transformations.translation_matrix(pose[0])
        M[:,3] = M_t[:,3]

        return M

    def get_postion(self,name):
        not_found = True
        while not_found:
            try:
                pose = self.tl.lookupTransform('map',name, rospy.Time(0))
                not_found = False
            except:
                rospy.logerr("waiting for pose")

        M = tf.transformations.quaternion_matrix(pose[1])
        M_t = tf.transformations.translation_matrix(pose[0])
        M[:,3] = M_t[:,3]

        return pose[0]




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

    def sample_corners(self):

        head_up = self.get_postion("head_up")
        head_down = self.get_postion("head_down")
        bottom_up = self.get_postion("bottom_up")
        bottom_down = self.get_postion("bottom_down")


        # Get true midpoints of table edges
        # (i.e. don't assume bottom_up and head_up have the same y value
        # in case sensor image is tilted)
        middle_up   =   np.array([(bottom_up[0] + head_up[0])/2, 
                    (bottom_up[1] + head_up[1])/2, 
                    bottom_down[2]])

        middle_down =   np.array([(bottom_down[0] + head_down[0])/2, 
                        (bottom_down[1] + head_down[1])/2, 
                        bottom_down[2]] )

        bottom_middle = np.array([(bottom_down[0] + bottom_up[0])/2, 
                        (bottom_down[1] + bottom_up[1])/2, 
                        bottom_down[2]] )

        head_middle =   np.array([(head_down[0] + head_up[0])/2, 
                        (head_down[1] + head_up[1])/2, 
                        bottom_down[2]])

        center  = np.array([(bottom_middle[0] + head_middle[0])/2,
                    (middle_down[1] + middle_up[1])/2,
                    bottom_down[2]])


        # Generate random point in sensor frame for the closer corner
        # Draws from gaussian distribution
        u_down = np.random.uniform(low=0.0,high=0.5)
        v_down = np.random.uniform(low=0.3,high=1.0)
        x_down = center[0] + u_down*LA.norm(center - bottom_middle)
        y_down = center[1] + v_down*LA.norm(center - middle_down)
        down_corner = (x_down, y_down, center[2])

        # Generate random point in sensor frame for the further corner
        # Draws from gaussian distribution
        u_up = np.random.uniform(low=0.0,high=0.5)
        v_up = np.random.uniform(low=0.3,high=1.0)
        x_up = center[0] - u_up*LA.norm(center - bottom_middle)
        y_up = center[1] - v_up*LA.norm(center - middle_up)
        up_corner = (x_up, y_up, center[2])    

        print("Here's the initial state sampled:")
        print "  CENTER ", center
        print "  UP CORNER ", up_corner
        print "  DOWN CORNER ", down_corner
        # Daniel: this is causing some pjust to see what happens here.
        #if center[1] < 0.0 or center[2] < 0.0:
        #    raise "ROBOT TRANSFROM INCORRECT"
        if center[1] < 0.0 or center[2] < 0.0:
            print("Warning: initial state `center` is not right; ignoring for now ...")
        print("")
        return down_corner, up_corner


    def sample_initial_state(self):
        down_corner, up_corner = self.sample_corners()
        button = 1.0
        while button > -0.1:
            control_state = self.xbox.getControllerState()
            d_pad = control_state['d_pad']
            button = d_pad[1]
            self.make_projection(down_corner,up_corner)
        return down_corner, up_corner


if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    whole_body = robot.get('whole_body')
    omni_base = robot.get('omni_base')
    com = COM()
    rgbd_map = RGBD2Map()
    cam = RGBD()
    com.go_to_initial_state(whole_body)
    tt = TableTop()
    tt.find_table(robot)
    # tt.move_to_pose(omni_base,'lower_start')
    # whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
    time.sleep(5)
    IS = InitialSampler(cam)

    while True:
        IS.sample_initial_state()
