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

from data_aug.draw_cross_hair import DrawPrediction

from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM

from il_ros_hsr.p_pi.bed_making.tensioner import Tensioner

from il_ros_hsr.core.sensors import Gripper_Torque
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
import thread


__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0


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
        self.tl = tf.TransformListener()
        self.gp = graspPlanner
        self.gripper = gripper
        self.options = options
        self.com = COM()

        self.tension = Tensioner()

    


    def compute_trans_to_map(self,norm_pose,rot):

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



    def loop_broadcast(self,norm_pose,rot,count):


        norm_pose,rot = self.compute_trans_to_map(norm_pose,rot)

        while True:
            self.br.sendTransform((norm_pose[0], norm_pose[1], norm_pose[2]),
                    #tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
                    rot,
                    rospy.Time.now(),
                    'bed_i_'+str(count),
                    #'head_rgbd_sensor_link')
                    'map')

            self.br.sendTransform((0.0, 0.0, -cfg.GRIPPER_HEIGHT),
                    tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
                    rospy.Time.now(),
                    'bed_'+str(count),
                    #'head_rgbd_sensor_link')
                    'bed_i_'+str(count))


    
    def broadcast_poses(self,poses,g_count):
        #while True: 
        
        count = 0

        
        for pose in poses:
            
            num_pose = pose[1]
            label = pose[0]

            

            td_points = self.pcm.projectPixelTo3dRay((num_pose[0],num_pose[1]))
            print "DE PROJECTED POINTS ",td_points
            norm_pose = np.array(td_points)
            norm_pose = norm_pose/norm_pose[2]
            norm_pose = norm_pose*(cfg.MM_TO_M*num_pose[2])
            print "NORMALIZED POINTS ",norm_pose
            
            #pose = np.array([td_points[0],td_points[1],0.001*num_pose[2]])
            a = tf.transformations.quaternion_from_euler(ai=-2.355,aj=-3.14,ak=0.0)
            b = tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=1.57)

            c = tf.transformations.quaternion_multiply(a,b)

            thread.start_new_thread(self.loop_broadcast,(norm_pose,c,g_count))
            time.sleep(0.3)
          
            count += 1



    def convert_crop(self,pose):

        pose[0] = self.options.OFFSET_Y + pose[0]
        pose[1] = self.options.OFFSET_X + pose[1]

        return pose

    def plot_on_true(self,pose,true_img):

        #pose = self.convert_crop(pose)

        dp = DrawPrediction()

        image = dp.draw_prediction(np.copy(true_img),pose)

        cv2.imshow('label_given',image)

        cv2.waitKey(30)



    def find_pick_region_labeler(self,results,c_img,d_img,count):
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

            depth = self.gp.find_mean_depth(d_img_c)

            poses.append([result['class'],[x,y,depth]])

        self.broadcast_poses(poses,count)

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
        whole_body.move_end_effector_pose(geometry.pose(),cards[0])

        self.com.grip_squeeze(self.gripper)
        
        self.tension.force_pull(whole_body,direction)
        self.com.grip_open(self.gripper)

        






if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()

    IPython.embed()