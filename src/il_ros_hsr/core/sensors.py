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
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, JointState


class RGBD(object):

    def __init__(self):
        topic_name_c = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        topic_name_i = '/hsrb/head_rgbd_sensor/rgb/camera_info'
        #topic_name_i = '/hsrb/head_rgbd_sensor/projector/camera_info'

        # Two possibilities for depth. Michael used `image_raw`, but HSR support
        # suggests `image_rect_raw`, but for that we need to patch NaNs.
        self.topic_name_d = topic_name_d = \
                '/hsrb/head_rgbd_sensor/depth_registered/image_raw'
        #self.topic_name_d = topic_name_d = \
        #        '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'

        self._bridge = CvBridge()
        self._input_color_image = None
        self._input_depth_image = None
        self._info = None
        self.is_updated = False

        # Subscribe color image data from HSR
        self._sub_color_image = rospy.Subscriber(
            topic_name_c, Image, self._color_image_cb)

        self._sub_depth_image = rospy.Subscriber(
            topic_name_d, Image, self._depth_image_cb)

        self._sub_info = rospy.Subscriber(
            topic_name_i, CameraInfo, self._info_cb)

        # Wait until connection
        #rospy.wait_for_message(topic_name_c, Image, timeout=1.0)
        #rospy.wait_for_message(topic_name_d, Image, timeout=100.0)

    def _color_image_cb(self, data):
        try:
            
            self._input_color_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
            self.color_time_stamped = data.header.stamp
            self.is_updated = True

        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def _depth_image_cb(self, data):
        try:
            self._input_depth_image = self._bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def _info_cb(self,data):
        try:
            self._info = data
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_color_data(self):
        return self._input_color_image

    def read_depth_data(self):
        dimg = self._input_depth_image
        # Sigh, this seems bugged ...
        #if 'image_rect_raw' in self.topic_name_d:
        #    cv2.patchNaNs(dimg, 0.0)
        return dimg

    def read_info_data(self):
        return self._info


class Joint_Positions(object):

    def __init__(self):
        topic_name = '/hsrb/joint_states'
        self._bridge = CvBridge()
        self._input_image = None
     
        # Subscribe color image data from HSR
        self._input_state = rospy.Subscriber(
            topic_name, JointState, self._state_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, WrenchStamped, timeout=5.0)

    def _state_cb(self, data):
        try:
            self._input_state = data
        except:
            rospy.logerr('could not read joint positions')

    def read_data(self):
        return self._input_state


class Gripper_Torque(object):

    def __init__(self):
        topic_name = '/hsrb/wrist_wrench/compensated'
        self._bridge = CvBridge()
        self._input_image = None

        self.record = False
        self.history = []
     
        # Subscribe color image data from HSR
        self._input_torque = rospy.Subscriber(
            topic_name, WrenchStamped, self._wrench_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, WrenchStamped, timeout=5.0)

    def _wrench_cb(self, data):
        try:
            self._input_torque = data
            if(self.record):
                self.history.append(data)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_data(self):
        return self._input_torque

    def save(self):
        self.record = True

    def stop(self):
        self.record = False

    def get_history(self):
        return self.history


class Wrist_RGB(object):

    def __init__(self):
        topic_name = '/hsrb/hand_camera/image_raw'
        self._bridge = CvBridge()
        self._input_image = None

        # Subscribe color image data from HSR
        self._image_sub = rospy.Subscriber(
            topic_name, Image, self._color_image_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, Image, timeout=5.0)

    def _color_image_cb(self, data):
        try:
            self._input_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_data(self):
        return self._input_image

class Head_Center_RGB(object):
    def __init__(self):
        topic_name = '/hsrb/head_center_camera/image_raw'
        self._bridge = CvBridge()
        self._input_image = None

        # Subscribe color image data from HSR
        self._image_sub = rospy.Subscriber(
            topic_name, Image, self._color_image_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, Image, timeout=5.0)

    def _color_image_cb(self, data):
        try:
            self._input_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_data(self):
        return self._input_image


class Head_Left_RGB(object):
    def __init__(self):
        topic_name = '/hsrb/head_l_stereo_camera/image_rect_color'
        topic_name_i = '/hsrb/head_l_stereo_camera/camera_info'

        self._bridge = CvBridge()
        self._input_image = None

        # Subscribe color image data from HSR
        self._image_sub = rospy.Subscriber(
            topic_name, Image, self._color_image_cb)

         # Subscribe color image data from HSR
        self._info_sub = rospy.Subscriber(
            topic_name_i, CameraInfo, self._info_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, Image, timeout=5.0)

    def _color_image_cb(self, data):
        try:
            self._input_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def _info_cb(self,data):
        try:
            self._info = data
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)


    def read_data(self):
        return self._input_image

    def read_info(self):
        return self._info

class Laser_Scan(object):

    def __init__(self):
        topic_name = '/hsrb/base_scan'
        
        # Subscribe color image data from HSR
        self._laser_scan = rospy.Subscriber(
            topic_name, LaserScan, self._wrench_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, WrenchStamped, timeout=5.0)

    def _laser_cb(self, data):
        try:
            self._laser_scan = data
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_data(self):
        return self._laser_scan


class Head_Right_RGB(object):
    def __init__(self):
        topic_name = '/hsrb/head_r_stereo_camera/image_rect_color'
        topic_name_i = '/hsrb/head_r_stereo_camera/camera_info'

        self._bridge = CvBridge()
        self._input_image = None

        # Subscribe color image data from HSR
        self._image_sub = rospy.Subscriber(
            topic_name, Image, self._color_image_cb)

         # Subscribe color image data from HSR
        self._info_sub = rospy.Subscriber(
            topic_name_i, CameraInfo, self._info_cb)
        # Wait until connection
        rospy.wait_for_message(topic_name, Image, timeout=5.0)

    def _color_image_cb(self, data):
        try:
            self._input_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def _info_cb(self,data):
        try:
            self._info = data
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)


    def read_data(self):
        return self._input_image

    def read_info(self):
        return self._info


if __name__=='__main__':

    # Prepare the listener
    eye = EyeHand()    
    torque = Torque()
    try:
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
    except:
        rospy.logerr('fail TransformListener')
        sys.exit()

    # Open the hand, and transit to suitable posture to move
    try:
        gripper.command(1.0)
        whole_body.move_to_go()
    except:
        rospy.logerr('fail to init')
        sys.exit()

    # # Move to location where the mug is viewable
    # try:
    #     omni_base.go(sofa_pos[0], sofa_pos[1], sofa_pos[2], _MOVE_TIMEOUT)
    # except:
    #     rospy.logerr('fail to move')
    #     sys.exit()

    # Recognize the mug and pick
    # while True: 
    #     print torque._input_torque

    try:
        # Transit to initial grasping posture
        whole_body.move_to_neutral()
        #whole_body.linear_weight = 100.0
        print "GOT HERE"
        # Move the hand to front of the mug
        #whole_body.end_effector_frame = u'hand_palm_link'
        while True:
            cv2.imshow('debug',eye._input_image)
            cv2.waitKey(30)
            #whole_body.move_end_effector_pose(geometry.pose(z=1.0), ref_frame_id='hand_palm_link')
            whole_body.move_end_effector_by_line((0, 1, 0), 0.1)
    except:
        rospy.logerr('fail to grasp')
        sys.exit()
