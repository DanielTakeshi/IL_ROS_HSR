import sys
sys.path.append('/opt/tmc/ros/indigo/lib/python2.7/dist-packages')
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
sys.path.append('/home/autolab/Workspaces/seita_working/fast_grasp_detect/')
from fast_grasp_detect.labelers.online_labeler import QueryLabeler
#from online_labeler import QueryLabeler
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


class BedMaker():

    def __init__(self):
        """For data collection of bed-making, not the deployment.
        Assumes we roll out the robot's policy via code (not via human touch).
        """
        self.robot = hsrb_interface.Robot()
        self.rgbd_map = RGBD2Map()
        self.omni_base = self.robot.get('omni_base')
        self.whole_body = self.robot.get('whole_body')
        self.cam = RGBD()
        self.com = COM()

        # Web interface for data labeling and inspection
        if cfg.USE_WEB_INTERFACE:
            self.wl = Web_Labeler()
        else:
            self.wl = Python_Labeler(cam = self.cam)

        # Set up initial state, table, etc.
        self.com.go_to_initial_state(self.whole_body)
        self.tt = TableTop()
        self.tt.find_table(self.robot)
        self.ins = InitialSampler(self.cam)
        self.side = 'BOTTOM'
        self.grasp_count = 0

        # Bells and whistles; note the 'success check' to check if transitioning
        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        self.gp = GraspPlanner()
        self.gripper = Bed_Gripper(self.gp,self.cam,self.com.Options,self.robot.get('gripper'))
        self.sc = Success_Check(self.whole_body,self.tt,self.cam,self.omni_base)

        self.ss = Self_Supervised(self.cam) # ignore
        time.sleep(4)
        print("Finished creating BedMaker()! Now adjust as follows:")

        self.sampling_type = ''
        if np.random.rand() < 0.5:
            if np.random.rand() < 0.5:
                print("Make the bed sheet CURVED, with LOTS OF WHITE VISIBLE")
                self.sampling_type = 'curved-white'
            else:
                print("Make the bed sheet CURVED, with not much white visible")
                self.sampling_type = 'curved-nowhite'
        else:
            if np.random.rand() < 0.5:
                print("Make the bed sheet (ROUGHLY) STRAIGHT, with LOTS OF WHITE VISIBLE")
                self.sampling_type = 'straight-white'
            else:
                print("Make the bed sheet (ROUGHLY) STRAIGHT, with not much white visible")
                self.sampling_type = 'straight-nowhite'


    def bed_make(self):
        """Runs the pipeline for data collection.
        You can run this for multiple bed-making trajectories.
        """
        self.rollout_data = []
        self.get_new_grasp = True

        # I think, creates red line in GUI where we adjust the bed to match it.
        if cfg.INS_SAMPLE:
            u_c, d_c = self.ins.sample_initial_state()
            self.rollout_data.append( [u_c, d_c, self.sampling_type] )

        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            if(not c_img.all() == None and not d_img.all() == None):
                if self.get_new_grasp:
                    self.position_head()

                    # Human supervisor labels. data = dictionary of relevant info
                    data = self.wl.label_image(c_img)
                    c_img = self.cam.read_color_data()
                    d_img = self.cam.read_depth_data()
                    self.add_data_point(c_img,d_img,data,self.side,'grasp')

                    # Broadcasts grasp pose
                    self.gripper.find_pick_region_labeler(data,c_img,d_img,self.grasp_count)

                    # Ignore
                    if cfg.SS_LEARN:
                        grasp_points = self.ss.learn(self.whole_body,self.grasp_count)
                        self.add_ss_data(grasp_points,data,self.side,'grasp')

                # Execute the grasp and check for success.
                pick_found,bed_pick = self.check_card_found()
                if self.side == "BOTTOM":
                    self.gripper.execute_grasp(bed_pick,self.whole_body,'head_down')
                else:
                    self.gripper.execute_grasp(bed_pick,self.whole_body,'head_up')
                self.check_success_state()


    def check_success_state(self):
        """
        Checks whether a single grasp in a bed-making trajectory succeeded.
        Depends on which side of the bed the HSR is at. Invokes human supervisor
        and transitions the HSR if successful.
        """
        if self.side == "BOTTOM":
            success, data = self.sc.check_bottom_success(self.wl)
        else:
            success, data = self.sc.check_top_success(self.wl)
        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()
        self.add_data_point(c_img,d_img,data,self.side,'success')
        print "WAS SUCCESFUL: "
        print success

        # Handle transitioning to different side
        if success:
            if cfg.SS_LEARN:
                grasp_points = self.ss.learn(self.whole_body,self.grasp_count)
                self.add_ss_data(grasp_points,data,self.side,'success')
            if self.side == "BOTTOM":
                self.transition_to_top()
            else:
                self.transition_to_start()
            self.update_side()
            self.grasp_count += 1
            self.get_new_grasp = True
        else:
            self.grasp_count += 1
            # If grasp failure, invokes finding region again and add new data
            self.gripper.find_pick_region_labeler(data,c_img,d_img,self.grasp_count)
            self.add_data_point(c_img,d_img,data,self.side,'grasp')
            self.get_new_grasp = False
            if cfg.SS_LEARN:
                grasp_points = self.ss.learn(self.whole_body,self.grasp_count)
                self.add_ss_data(grasp_points,data,self.side,'success')


    def update_side(self):
        """TODO: extend to multiple side switches?"""
        if self.side == "BOTTOM":
            self.side = "TOP"


    def transition_to_top(self):
        """Transition to top (not bottom)."""
        if cfg.DEBUG_MODE:
            self.com.save_rollout(self.rollout_data)
            self.tt.move_to_pose(self.omni_base,'lower_mid')
            sys.exit()
        else:
            self.move_to_top_side()


    def transition_to_start(self):
        """Transition to start=bottom, save rollout data, exit program.
        Saves to a supervisor's directory since we're using a supervisor.
        """
        self.com.save_rollout(self.rollout_data)
        self.move_to_start()
        sys.exit()


    def add_data_point(self,c_img,d_img,data,side,typ,pose = None):
        """Adds a dictionary to the `rollout_data` list."""
        grasp_point = {}
        grasp_point['c_img'] = c_img
        grasp_point['d_img'] = d_img
        if pose == None:
            label = data['objects'][0]['box']
            pose = [(label[2]-label[0])/2.0+label[0],(label[3]-label[1])/2.0+label[1]]
        grasp_point['pose'] = pose
        grasp_point['class'] = data['objects'][0]['class']
        grasp_point['side'] = side
        grasp_point['type'] = typ
        self.rollout_data.append(grasp_point)


    def position_head(self):
        if self.side == "TOP":
            self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        elif self.side == "BOTTOM":
            self.tt.move_to_pose(self.omni_base,'lower_start')
            self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})


    def add_ss_data(self,g_points,data,side,typ):
        for g_point in g_points:
            self.add_data_point(g_point['c_img'],g_point['d_img'],data,side,typ,pose=g_point['pose'])


    def move_to_top_side(self):
        self.tt.move_to_pose(self.omni_base,'right_down')
        #self.tt.move_to_pose(self.omni_base,'right_mid')
        self.tt.move_to_pose(self.omni_base,'right_up')
        self.tt.move_to_pose(self.omni_base,'top_mid')


    def move_to_start(self):
        self.tt.move_to_pose(self.omni_base,'right_up')
        #self.tt.move_to_pose(self.omni_base,'right_mid')
        self.tt.move_to_pose(self.omni_base,'right_down')
        self.tt.move_to_pose(self.omni_base,'lower_mid')
        

    def check_bottom_success(self):
        """TODO we call the success check method ... this never called?"""
        self.tt.move_to_pose(self.omni_base,'lower_mid')
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})


    def check_card_found(self):
        """Looks up the pose for where the HSR's hand should go to."""
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


    def find_mean_depth(self,d_img):
        """TODO never called?"""
        indx = np.nonzero(d_img)
        mean = np.mean(d_img[indx])
        return


if __name__ == "__main__":
    cp = BedMaker()
    cp.bed_make()
