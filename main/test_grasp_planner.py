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
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')

from online_labeler import QueryLabeler
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler

from il_ros_hsr.p_pi.bed_making.check_success import Success_Check
from il_ros_hsr.p_pi.bed_making.get_success import get_success
from il_ros_hsr.p_pi.bed_making.self_supervised import Self_Supervised
import il_ros_hsr.p_pi.bed_making.config_bed as cfg


from il_ros_hsr.core.rgbd_to_map import RGBD2Map

from il_ros_hsr.p_pi.bed_making.initial_state_sampler import InitialSampler
class GraspTester():

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
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
      

        self.cam = RGBD()
        


        self.wl = Python_Labeler(cam = self.cam)

        

        self.gp = GraspPlanner()
        
    def test_grasper(self):
        c_img = self.cam.read_color_data()
        data = self.wl.label_image(c_img)

        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()

        point = self.get_points(data)

        self.gp.compute_grasp(point,d_img,c_img)


    def get_points(self,data):
        
        result = data['objects'][0]

        x_min = float(result['box'][0])
        y_min = float(result['box'][1])
        x_max = float(result['box'][2])
        y_max = float(result['box'][3])

        p_0 = (x_min,y_min)
        p_1 = (x_max,y_max)

        return (p_0,p_1)


if __name__ == "__main__":
   
    
    cp = GraspTester()

    cp.test_grasper()

