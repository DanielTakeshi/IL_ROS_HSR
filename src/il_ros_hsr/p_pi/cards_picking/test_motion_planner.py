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
from il_ros_hsr.core.joystick_X import  JoyStick_X
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
from yolo.detector_fast import Detector
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

        self.whole_body = self.robot.get('whole_body')
        

        #self.cam = RGBD()
        self.com = COM()

        self.com.go_to_initial_state(self.whole_body)
       

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        self.gp = GraspPlanner()
        #self.detector = Detector()

        self.joystick = JoyStick_X(self.com)

        #self.suction = Suction(self.gp,self.cam,self.com.Options)

        #self.suction.stop()
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
                
            cur_recording = self.joystick.get_record_actions_passive()
            self.broadcast_transform()
                
            
            if(cur_recording[0] < -0.1):
                
                
                self.go_to_centroid(self.whole_body)

                    #self.com.go_to_initial_state(self.whole_body)


    def broadcast_transform(self):

        try:
            self.br.sendTransform((0.0,0.0,-0.02),
                    tf.transformations.quaternion_from_euler(ai=-0.785,aj=0.0,ak=0.0),
                    rospy.Time.now(),
                    'transform_ar_marker',
                    'ar_marker/11')
        except:
            rospy.logerr('ar marker not found')


    def go_to_centroid(self,whole_body):

        
        whole_body.end_effector_frame = 'hand_l_finger_vacuum_frame'
        nothing = True
        
        #self.whole_body.move_to_neutral()

        whole_body.move_end_effector_pose(geometry.pose(z = -0.02,ei = -0.785),'ar_marker/11')
        #whole_body.move_end_effector_by_line((0,0,1),0.02)
        #self.start()

        #whole_body.move_to_joint_positions({'arm_lift_joint':0.23})

       



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
        return False, []

        




if __name__ == "__main__":
   
    
    cp = CardPicker()

    cp.card_pick()

