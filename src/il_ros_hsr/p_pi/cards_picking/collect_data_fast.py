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
from il_ros_hsr.core.suction import Suction
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import tf
import rospy

from il_ros_hsr.core.grasp_planner import GraspPlanner

from il_ros_hsr.p_pi.cards_picking.com import Cards_COM as COM
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
from yolo.detector import Detector
import yolo.config_card as cfg

from online_labeler import QueryLabeler
from image_geometry import PinholeCameraModel as PCM
from il_ros_hsr.core.joystick_X import  JoyStick_X


class CardPicker():

    def __init__(self):
        '''
        Initialization class for a Policy

        Parameters
        ----------
        yumi : An instianted yumi robot 
        com : The common class for the robot
        cam : An open bincam class

        debug : bool 

            A bool to indicate whether or not to display a training set point for 
            debuging. 

        '''

        self.robot = hsrb_interface.Robot()

        self.omni_base = self.robot.get('omni_base')
        self.whole_body = self.robot.get('whole_body')
        

        self.cam = RGBD()
        self.com = COM()

        #self.com.go_to_initial_state(self.whole_body)

        self.count = 425

        self.joystick = JoyStick_X(self.com)
        self.true_count = 0

       


    def collect_data(self):

        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            c_img,d_img = self.com.format_data(c_img,d_img)


            cv2.imshow('video_feed',c_img)
            cv2.waitKey(30)

            cur_recording = self.joystick.get_record_actions_passive()
            if(cur_recording[0] < -0.1 and self.true_count%5 == 0):
                print "PHOTO SNAPPED " + str(self.count)
                cv2.imwrite(cfg.IMAGE_PATH + '/frame_'+str(self.count)+'.png',c_img)
                self.count += 1
            self.true_count += 1





if __name__ == "__main__":
   
    
    cp = CardPicker()

    cp.collect_data()

