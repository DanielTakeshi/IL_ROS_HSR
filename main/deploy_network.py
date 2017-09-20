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
import time

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

from il_ros_hsr.p_pi.bed_making.net_success import Success_Net
from il_ros_hsr.p_pi.bed_making.self_supervised import Self_Supervised
import il_ros_hsr.p_pi.bed_making.config_bed as cfg


from il_ros_hsr.core.rgbd_to_map import RGBD2Map
from detectors.grasp_detector import GDetector
from il_ros_hsr.p_pi.bed_making.initial_state_sampler import InitialSampler
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
        self.rgbd_map = RGBD2Map()

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
        self.ins = InitialSampler(self.cam)
    
        self.grasp_count = 0
      

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()

        self.gp = GraspPlanner()

        self.gripper = Bed_Gripper(self.gp,self.cam,self.com.Options,self.robot.get('gripper'))

        self.g_detector = GDetector(cfg.GRASP_NET_NAME)

        self.sn = Success_Net(self.whole_body,self.tt,self.cam,self.omni_base)

        c_img = self.cam.read_color_data()

        self.sn.sdect.predict(c_img)
        

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


    def bed_make(self):

        self.rollout_stats = []
        self.get_new_grasp = True

        if cfg.INS_SAMPLE:
                u_c,d_c = self.ins.sample_initial_state()

                self.rollout_stats.append([u_c,d_c])

        self.new_grasp = True
        while True:

            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            if(not c_img == None and not d_img == None):

                if self.new_grasp:
                    self.position_head()
                else: 
                    self.new_grasp = True
                time.sleep(3)
                
                c_img = self.cam.read_color_data()
                d_img = self.cam.read_depth_data()


                sgraspt = time.time()
                data = self.g_detector.predict(np.copy(c_img))
                egraspt = time.time()
          
                print("Grasp predict time: " + str(egraspt - sgraspt))
                
                self.record_stats(c_img,d_img,data,self.side,'grasp')
                    

                
                
                self.gripper.find_pick_region_net(data,c_img,d_img,self.grasp_count)
                


            
                pick_found,bed_pick = self.check_card_found()

                if self.side == "BOTTOM":
                    self.gripper.execute_grasp(bed_pick,self.whole_body,'head_down')
                else:
                    self.gripper.execute_grasp(bed_pick,self.whole_body,'head_up')

                self.check_success_state(c_img,d_img)
                           
               
    def check_success_state(self,c_img,d_img):

        
        if self.side == "BOTTOM":
            success, data, c_img  = self.sn.check_bottom_success(self.wl)
        else: 
            success, data, c_img = self.sn.check_top_success(self.wl)

     
        self.record_stats(c_img,d_img,data,self.side,'success')

       
        print "WAS SUCCESFUL: "
        print success
        if(success):

            if self.side == "BOTTOM":
                self.transition_to_top()
            else:
                self.transition_to_start()

            self.update_side()

        else: 
            self.new_grasp = False

        self.grasp_count += 1

        if self.grasp_count > cfg.GRASP_OUT: 
            self.transition_to_start()
           
   
    def update_side(self):

        if self.side == "BOTTOM":
            self.side = "TOP"

    def transition_to_top(self):
        if cfg.DEBUG_MODE:
            self.com.save_stat(self.rollout_stats)
            self.tt.move_to_pose(self.omni_base,'lower_mid')
            sys.exit()
        else:
            self.move_to_top_side()

    def transition_to_start(self):
        self.com.save_stat(self.rollout_stats)
        self.move_to_start()
        sys.exit()

    def record_stats(self,c_img,d_img,data,side,typ):

        grasp_point = {}

        grasp_point['c_img'] = c_img
        grasp_point['d_img'] = d_img
        
        if typ == "grasp":
            grasp_point['net_pose'] = data
        else: 
            grasp_point['net_trans'] = data

        grasp_point['side'] = side
        grasp_point['type'] = typ

        self.rollout_stats.append(grasp_point)

    def position_head(self):

        if self.side == "TOP":
            self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        elif self.side == "BOTTOM":
            self.tt.move_to_pose(self.omni_base,'lower_start')
            self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})



    def move_to_top_side(self):

        self.tt.move_to_pose(self.omni_base,'right_down')
      
    
        self.tt.move_to_pose(self.omni_base,'right_up')
        
        self.tt.move_to_pose(self.omni_base,'top_mid')


    def move_to_start(self):

        if self.side == "BOTTOM":
            self.tt.move_to_pose(self.omni_base,'lower_mid')
        else:

            self.tt.move_to_pose(self.omni_base,'right_up')
           
            self.tt.move_to_pose(self.omni_base,'right_down')
            self.tt.move_to_pose(self.omni_base,'lower_mid')
    

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
                    f_p = self.tl.lookupTransform('map',transform, rospy.Time(0))
                    cards.append(transform)

        except: 
            rospy.logerr('bed pick not found yet')
                

        return True, cards
 

        




if __name__ == "__main__":
   
    
    cp = BedMaker()
    
    cp.bed_make()

