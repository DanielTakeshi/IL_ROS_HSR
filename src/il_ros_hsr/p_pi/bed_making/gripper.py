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
from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel as PCM
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
from il_ros_hsr.p_pi.bed_making.tensioner import Tensioner
from il_ros_hsr.core.sensors import Gripper_Torque
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
import thread
from  numpy.random import multivariate_normal as mvn
__SUCTION_TIMEOUT__ = rospy.Duration(20.0)
_CONNECTION_TIMEOUT = 10.0


class Bed_Gripper(object):
    """
    Handles the gripper for bed-making, similar to CraneGripper since it creates
    poses for the robot's end-effector to go to, except for functionality
    related to the neural networks and python labelers.
    """

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
        # Bells and whisltes
        self.pcm = PCM()
        self.pcm.fromCameraInfo(cam_info)
        self.br = tf.TransformBroadcaster()
        self.tl = tf.TransformListener()
        self.gp = graspPlanner
        self.gripper = gripper
        self.options = options
        self.com = COM()

        # See paper for details, used to release grasp if too much force.
        self.tension = Tensioner()

        # Side, need to change this.
        self.side = 'BOTTOM'
        self.offset_x = 0.0 # positive means going to other side of bed (both sides!)
        self.offset_z = 0.0 # negative means going UP to ceiling
        self.apply_offset = False


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


    def loop_broadcast(self, norm_pose, rot, count):
        norm_pose, rot = self.compute_trans_to_map(norm_pose, rot)
        gripper_height = cfg.GRIPPER_HEIGHT

        # Bleh ... :-(. Pretend gripper height is shorter.  UPDATE: let's not do
        # this. It's a bad idea. Put tape under table legs for better balance.
        #if self.side == 'TOP':
        #    gripper_height -= 0.015

        # But we may want some extra offset in the x and z directions.
        if self.apply_offset:
            print("self.apply_offset = True")
            self.offset_x = 0.010
            self.offset_z = 0.010
        else:
            print("self.apply_offset = False")
            self.offset_x = 0.0
            self.offset_z = 0.0

        while True:
            self.br.sendTransform((norm_pose[0], norm_pose[1], norm_pose[2]),
                    #tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
                    rot,
                    rospy.Time.now(),
                    'bed_i_'+str(count),
                    #'head_rgbd_sensor_link')
                    'map')
            self.br.sendTransform((self.offset_x, 0.0, -gripper_height + self.offset_z),
                    tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=0.0),
                    rospy.Time.now(),
                    'bed_'+str(count),
                    #'head_rgbd_sensor_link')
                    'bed_i_'+str(count))

    
    def broadcast_poses(self,poses,g_count):
        #while True: 
        count = 0
        for pose in poses:
            num_pose = pose[1] # this is [x,y,depth]
            label = pose[0]
            td_points = self.pcm.projectPixelTo3dRay((num_pose[0],num_pose[1]))
            print("\nIn `bed_making.gripper.broadcast_poses()`")
            print("  DE PROJECTED POINTS {}".format(td_points))
            norm_pose = np.array(td_points)
            norm_pose = norm_pose/norm_pose[2]
            norm_pose = norm_pose*(cfg.MM_TO_M*num_pose[2])
            print("  NORMALIZED POINTS {}\n".format(norm_pose))
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


    def plot_on_true(self, pose, true_img):
        """Another debug helper method, shows the img with cross hair."""
        #pose = self.convert_crop(pose)
        dp = DrawPrediction()
        image = dp.draw_prediction(np.copy(true_img), pose)
        cv2.imshow('label_given',image)
        cv2.waitKey(30)
       

    def find_pick_region_net(self, pose, c_img, d_img, count, side, apply_offset=None):
        """Called during bed-making deployment w/neural network, creates a pose.

        It relies on the raw depth image! DO NOT PASS A PROCESSED DEPTH IMAGE!!!
        Also, shows the image to the user via `plot_on_true`.

        Update: pass in the 'side' as well, because I am getting some weird
        cases where the plane formed by the four table 'corner' frames (head up,
        head down, bottom up, bottom down) seem to be slightly at an angle. Ugh.
        So pretend the gripper height has 'decreased' by 1cm for the other side
        of the bed.

        Args:
            pose: (x,y) point as derived from the grasp detector network
        """
        assert side in ['BOTTOM', 'TOP']
        self.side = side
        self.apply_offset = apply_offset

        poses = []
        p_list = []
        x,y = pose
        print("in bed_making.gripper, PREDICTION {}".format(pose))
        self.plot_on_true([x,y], c_img)

        #Crop D+img
        d_img_c = d_img[int(y-cfg.BOX) : int(y+cfg.BOX) , int(x-cfg.BOX) : int(cfg.BOX+x)]

        depth = self.gp.find_mean_depth(d_img_c)
        poses.append( [1.0, [x,y,depth]] )
        self.broadcast_poses(poses, count)


    def find_pick_region_labeler(self, results, c_img, d_img, count):
        """Called during bed-making data collection, only if we use the labeler.

        It relies on the raw depth image! DO NOT PASS A PROCESSED DEPTH IMAGE!!!
        Also, shows the image to the user via `plot_on_true`.

        NOTE: main difference between this and the other method for the net is
        that the `results` argument encodes the pose implicitly and we have to
        compute it. Otherwise, though, the depth computation is the same, and
        cropping is the same, so the methods do similar stuff.

        Args:
            results: Dict from QueryLabeler class (human supervisor).
        """
        poses = []
        p_list = []

        for result in results['objects']:
            print result
            x_min = float(result['box'][0])
            y_min = float(result['box'][1])
            x_max = float(result['box'][2])
            y_max = float(result['box'][3])
            x = (x_max-x_min)/2.0 + x_min
            y = (y_max - y_min)/2.0 + y_min

            # Will need to test; I assume this requires human intervention.
            if cfg.USE_DART:
                pose = np.array([x,y])
                action_rand = mvn(pose,cfg.DART_MAT)
                print "ACTION RAND ",action_rand
                print "POSE ", pose
                x = action_rand[0]
                y = action_rand[1]
            self.plot_on_true([x,y],c_img)

            #Crop D+img
            d_img_c = d_img[int(y-cfg.BOX) : int(y+cfg.BOX) , int(x-cfg.BOX) : int(cfg.BOX+x)]
            depth = self.gp.find_mean_depth(d_img_c)
            # Note that `result['class']` is an integer (a class index).
            # 0=success, anything else indicates a grasping failure.
            poses.append([result['class'],[x,y,depth]])
        self.broadcast_poses(poses,count)


    def find_pick_region(self,results,c_img,d_img):
        """ From suctioning code, not the bed-making. Ignore it. """
        poses = []
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


    def execute_grasp(self, cards, whole_body, direction):
        """ Executes grasp. Move to pose, squeeze, pull (w/tension), open. """
        whole_body.end_effector_frame = 'hand_palm_link'

        # Hmmm ... might help with frequent table bumping? Higher = more arm movement.
        whole_body.linear_weight = 60.0

        whole_body.move_end_effector_pose(geometry.pose(),cards[0])
        self.com.grip_squeeze(self.gripper)

        # Then after we grip, go back to the default value.
        whole_body.linear_weight = 3.0

        # Then we pull.
        self.tension.force_pull(whole_body,direction)
        self.com.grip_open(self.gripper)



if __name__=='__main__':
    robot =  hsrb_interface.Robot()
    suc = Suction()
    IPython.embed()
