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
import tf
import rospy

from il_ros_hsr.core.grasp_planner import GraspPlanner

from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
from yolo.detector import Detector
from image_geometry import PinholeCameraModel as PCM


class YoloDetect():

    def __init__(self,options,name = None):
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

        
        if(name == None):
            name = '07_14_15_27_17save.ckpt-12000'
        self.cam = RGBD()
        not_read = True
        while not_read:

            try:
                cam_info = self.cam.read_info_data()
                if(not cam_info == None):
                    not_read = False
            except:
                rospy.logerr('info not recieved')
       

        self.pcm = PCM()
        self.pcm.fromCameraInfo(cam_info)
        self.options = options
        self.detect = Detector(name)
        self.br = tf.TransformBroadcaster()
        self.gp = GraspPlanner()


    def check_depth(self,p_list,d_img):

        w,h = d_img.shape

        color_img = np.zeros([w,h,3])

        color_img[:,:,0] = d_img*(255.0/float(np.max(d_img)))

        for p in p_list:
            print p
            color_img[p[1]-5:p[1]+5,p[0]-5:5+p[0],1] = 255.0

        cv2.imshow('debug',color_img)
        cv2.waitKey(30)

    def get_detect(self,c_img,d_img):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''


        results = self.detect.numpy_detector(c_img)

        self.poses = []
        #IPython.embed()
        
        p_list = []
        for result in results:
            print result

            x = int(result['box'][0])
            y = int(result['box'][1])
            w = int(result['box'][2] / 2)
            h = int(result['box'][3] / 2)

            p_list.append([x,y])

            #Crop D+img
            d_img_c = d_img[y-h:y+h,x-w,:x+w]
            depth = self.gp.find_mean_depth(d_img_c)
            self.poses.append([result['class'],self.convert_crop([x,y,depth])])

        self.check_depth(p_list,d_img)

    def convert_crop(self,pose):

        pose[0] = self.options.OFFSET_Y + pose[0]
        pose[1] = self.options.OFFSET_X + pose[1]

        return pose
            


    def broadcast_poses(self):
        #while True: 
        poses = self.poses
        count = 0
        for pose in poses:
            
            num_pose = pose[1]
            label = pose[0]

            td_points = self.pcm.projectPixelTo3dRay((num_pose[0],num_pose[1]))
            pose = np.array([td_points[0],td_points[1],0.001*num_pose[2]])
            

            self.br.sendTransform((td_points[0], td_points[1], pose[2]),
                    (0.0, 0.0, 0.0, 1.0),
                    rospy.Time.now(),
                    label,
                    'head_rgbd_sensor_rgb_frame')
            count += 1




if __name__ == "__main__":
   
    
    yolo = YoloDetect()


