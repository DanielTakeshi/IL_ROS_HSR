'''
Policy wrapper class 

Author: Michael Laskey
'''
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
import rospy

from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM




class  Policy():

    def __init__(self,com,features):
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


        self.pubTwist = rospy.Publisher('/hsrb/command_velocity',
                          Twist, queue_size=1)

        self.com = com

        self.cam = RGBD()

        self.trajectory = []
        self.features = features


    def rollout(self):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class


        '''
        
        c_img = self.cam.read_color_data()
        
        pos = self.com.eval_policy(c_img,self.features)
        print pos
        twist = self.com.format_twist(pos)

        self.pubTwist.publish(twist)


if __name__ == "__main__":
   
    
    yumi = YuMiRobot()
    options = Options()
    com = COM(train=False)
    bincam = BinaryCamera(options)
    yumi.set_z('fine')
    
    
    bincam.open(threshTolerance= options.THRESH_TOLERANCE)

    frame = bincam.display_frame()

    yumi.set_v(1500)
    
    debug_overlay(bincam,options.binaries_dir+'rollout0_frame_0.jpg')
    
    
    pi = Policy(yumi,com,bincam=bincam)

    while True:
        pi.rollout()

    print "Done."
