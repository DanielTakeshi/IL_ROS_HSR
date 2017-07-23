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
from online_labeler import QueryLabeler
from image_geometry import PinholeCameraModel as PCM


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

        self.com.go_to_initial_state(self.whole_body)
       

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        self.gp = GraspPlanner()

        self.suction = Suction(self.gp,self.cam)
        
        #thread.start_new_thread(self.ql.run,())
        print "after thread"

       


    def find_mean_depth(self,d_img):
        '''
        Evaluates the current policy and then executes the motion 
        specified in the the common class
        '''

        indx = np.nonzero(d_img)

        mean = np.mean(d_img[indx])

        return


    def card_pick(self):

        while True:

            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            if(not c_img == None and not d_img == None):

                self.ql = QueryLabeler()
                self.ql.run(c_img)
                data = self.ql.label_data
                del self.ql


                self.suction.find_to_pick_region(data,c_img,d_img)
                
                card_found,cards = self.check_card_found()

                

                if(card_found):
                    self.suction.execute_grasp(cards,self.whole_body)

                self.com.go_to_initial_state(self.whole_body)


    def check_card_found(self):

        # try:
        transforms = self.tl.getFrameStrings()
    
        cards = []
        
        for transform in transforms:
            print transform
            if 'card' in transform:
                print 'got here'
                f_p = self.tl.lookupTransform('head_rgbd_sensor_rgb_frame',transform, rospy.Time(0))
                cards.append(transform)
                

        return True, cards
        # except: 
        #     return False, []

        




if __name__ == "__main__":
   
    
    cp = CardPicker()

    cp.card_pick()

