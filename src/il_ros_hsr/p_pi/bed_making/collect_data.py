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
from yolo.detector import Detector
from online_labeler import QueryLabeler
from image_geometry import PinholeCameraModel as PCM

from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler
from il_ros_hsr.p_pi.bed_making.check_success import Success_Check
import il_ros_hsr.p_pi.bed_making.config_bed as cfg

class BedMaker():

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

        
        self.side = 'BOTTOM'

        self.cam = RGBD()
        self.com = COM()

        if cfg.USE_WEB_INTERFACE:
            self.wl = Web_Labeler()
        else:
            self.wl = Python_Labeler(self.cam)

        self.com.go_to_initial_state(self.whole_body)
        

        self.tt = TableTop()
        self.tt.find_table(self.robot)

    
        self.grasp_count = 0
       

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        self.gp = GraspPlanner()

        self.gripper = Bed_Gripper(self.gp,self.cam,self.com.Options,self.robot.get('gripper'))

        self.sc = Success_Check(self.whole_body,self.tt,self.cam,self.omni_base)

        #self.test_current_point()
        time.sleep(4)
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


    def bed_pick(self):

        while True:

            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            if(not c_img == None and not d_img == None):

                

                c_img = self.cam.read_color_data()
                d_img = self.cam.read_depth_data()

                data = self.wl.label_image(c_img)
                
                self.gripper.find_pick_region_labeler(data,c_img,d_img,self.grasp_count)
                
                pick_found,bed_pick = self.check_card_found()

                self.grasp_count += 1

                if(pick_found):
                    if(self.side == 'BOTTOM'):
                        self.gripper.execute_grasp(bed_pick,self.whole_body,'head_down')
                        success = self.sc.check_bottom_success(self.wl)

                        print "WAS SUCCESFUL: "
                        print success
                        if(success):
                            self.move_to_top_side()
                            self.side = "TOP"

                    elif(self.side == "TOP"):
                        self.gripper.execute_grasp(bed_pick,self.whole_body,'head_up')
                        success = self.sc.check_top_success(self.wl)

                        print "WAS SUCCESFUL: "
                        print success

                        if(success):
                            self.side == "PILLOW"


    def test_current_point(self):

        self.gripper.tension.force_pull(self.whole_body,(0,1,0))
        self.gripper.com.grip_open(self.gripper)
        self.move_to_top_side()

    def move_to_top_side(self):

        self.tt.move_to_pose(self.omni_base,'right_down')
        self.tt.move_to_pose(self.omni_base,'right_mid')
    
        self.tt.move_to_pose(self.omni_base,'right_up')
        
        self.tt.move_to_pose(self.omni_base,'top_mid')




    def check_bottom_success(self):

        self.tt.move_to_pose(self.omni_base,'lower_mid')
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})



    def check_card_found(self):

        # try:
        transforms = self.tl.getFrameStrings()
    
        cards = []

        try:
        
            for transform in transforms:
                print transform
                current_grasp = 'bed_'+str(self.grasp_count)
                if current_grasp in transform:
                    print 'got here'
                    f_p = self.tl.lookupTransform('head_rgbd_sensor_rgb_frame',transform, rospy.Time(0))
                    cards.append(transform)

        except: 
            rospy.logerr('bed pick not found yet')
                

        return True, cards
 

        




if __name__ == "__main__":
   
    
    cp = BedMaker()
    
    cp.bed_pick()

