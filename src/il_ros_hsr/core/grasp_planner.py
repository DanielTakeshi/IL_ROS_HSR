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
from il_ros_hsr.core.rgbd_project import RGBD_Project
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')

from image_geometry import PinholeCameraModel as PCM


class GraspPlanner():

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

        NUM_GRASPS = 40
        GRIPPER_WIDTH = 40 #MM

        self.cam_project = RGBD_Project()


    def find_mean_depth(self,d_img):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''

        indx = np.nonzero(d_img)

        mean = np.mean(d_img[indx])

        return mean

    def find_max_depth(self,d_img):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''

        indx = np.nonzero(d_img)

        mean = np.max(d_img[indx])

        return mean

    def get_deprojected_points(self,points,d_img,c_img):
        p_0 = points[0]
        p_1 = points[1]

        points = []
        for x in range(int(p_0[0]),int(p_1[0])):
            for y in range(int(p_0[1]),int(p_1[1])):

                z = d_img[x,y]

                if z > 0.0:
                    pose = self.cam_project.deproject_pose((x,y,z))

                points.append(pose)

        return points

    def compute_grasp(self,points,d_img,c_img):

        points = self.get_deprojected_points(points,d_img,c_img)

        IPython.embed()
        









if __name__ == "__main__":
   
    
    yolo = YoloDetect()

