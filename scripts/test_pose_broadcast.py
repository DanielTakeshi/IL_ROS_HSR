''''
    Used to train a nerual network that maps an image to robot pose (x,y,z)
    Supports the Option of Synthetically Growing the data for Rotation and Translation 

    Author: Michael Laskey 


    FlagsO
    ----------
    --first (-f) : int
        The first roll out to be trained on (required)

    --last (-l) : int
        The last rollout to be trained on (reguired)

    --net_name (-n) : string
        The name of the network if their exists multiple nets for the task (not required)

    --demonstrator (-d) : string 
        The name of the person who trained the network (not required)

'''


import sys, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from hsrb_interface import geometry
import hsrb_interface

import IPython
from il_ros_hsr.tensor import inputdata
import numpy as np, argparse
import cPickle as pickle
import cv2
from skvideo.io import vwrite
from il_ros_hsr.core.yolo_detector import YoloDetect
from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions


#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM
import rospy
import time
#specific: fetches specific net file
#from deep_lfd.tensor.nets.net_grasp import Net_Grasp as Net 
########################################################


if __name__ == '__main__':

    
    robot = hsrb_interface.Robot()

    Options = options()
    yolo_detect = YoloDetect(Options)
    com = COM()

    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')
    gripper = robot.get('gripper')

    #com.go_to_initial_state(whole_body,gripper)
    cam = RGBD()
    time.sleep(3)

    while True:
        c_img = cam.read_color_data()
        d_img = cam.read_depth_data()



        c_img,d_img = com.format_data(c_img,d_img)

        yolo_detect.get_detect(c_img,d_img)

        yolo_detect.broadcast_poses()
           
            
           
    

       
    
