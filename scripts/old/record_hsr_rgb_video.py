from hsrb_interface import geometry
import hsrb_interface
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
import geometry_msgs
import controller_manager_msgs.srv
import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython
from numpy.random import normal
import time
#import listener
import thread

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import tf
import rospy

from il_ros_hsr.core.grasp_planner import GraspPlanner

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/fast_grasp_detect/')


from skvideo.io import vwrite
from il_ros_hsr.core.joystick_X import  JoyStick_X




if __name__ == '__main__':

    robot = hsrb_interface.Robot()
    videos_color = []
    cam = RGBD()

    jy = JoyStick_X()
    
    while True:

        time.sleep(0.05)
        img = cam.read_color_data()
        if(not img == None):
            cv2.imshow('debug2',img)
            cv2.waitKey(30)
            img = img[:, :, (2, 1, 0)]
            videos_color.append(img)

            cur_recording = jy.get_record_actions_passive()
            if(cur_recording[0] < -0.1):
                print "SAVING VIDEO"
              
                vwrite('bed_rgbd.mp4',videos_color)

    
