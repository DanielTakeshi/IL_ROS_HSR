import sys
sys.path.append('/opt/tmc/ros/indigo/lib/python2.7/dist-packages')
from hsrb_interface import geometry
import hsrb_interface
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
import geometry_msgs
import controller_manager_msgs.srv
from cv_bridge import CvBridge, CvBridgeError
import IPython
from numpy.random import normal
import cv2, time, thread, rospy, glob
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import tf
from il_ros_hsr.core.grasp_planner import GraspPlanner
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
sys.path.append('/home/autolab/Workspaces/michael_working/fast_grasp_detect/')
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
from il_ros_hsr.core.joystick_X import  JoyStick_X


class BedMaker():

    def __init__(self):
        """ For data collection where we manually simulate it by moving the bed
        with our hands.
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

        # PARAMETERS TO CHANGE 
        self.side = 'TOP'
        self.r_count = 0
        self.grasp_count = 0
        self.success_count = 0
        self.true_count = 0
        self.grasp = True
        self.r_count = self.get_rollout_number()
        self.joystick = JoyStick_X(self.com)

        # Set up initial state, table, etc.
        self.com.go_to_initial_state(self.whole_body)
        self.tt = TableTop()
        self.tt.find_table(self.robot)
        self.position_head()
        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        #self.test_current_point()
        time.sleep(4)
        #thread.start_new_thread(self.ql.run,())


    def get_rollout_number(self):
        if self.side == "BOTTOM":
            rollouts = glob.glob(cfg.FAST_PATH+'b_grasp/*.png')
        else:
            rollouts = glob.glob(cfg.FAST_PATH+'t_grasp/*.png')
        r_nums = []
        for r in rollouts:
            a = r[56:]
            i = a.find('_')
            r_num = int(a[:i])
            r_nums.append(r_num)
        return max(r_nums)+1


    def position_head(self):
        if self.side == "TOP":
            self.tt.move_to_pose(self.omni_base,'right_down')
            self.tt.move_to_pose(self.omni_base,'right_up')
            self.tt.move_to_pose(self.omni_base,'top_mid')
            self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        elif self.side == "BOTTOM":
            self.tt.move_to_pose(self.omni_base,'lower_start')
            self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})


    def collect_data_bed(self):
        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            cv2.imshow('video_feed',c_img)
            cv2.waitKey(30)
            cur_recording = self.joystick.get_record_actions_passive()
            if(cur_recording[0] < -0.1 and self.true_count%20 == 0):
                print "PHOTO SNAPPED "
                self.save_image(c_img)

                if self.grasp:
                    self.grasp_count += 1
                    self.grasp = False
                else:
                    self.success_count += 1
                    self.grasp = True

            if(cur_recording[1] < -0.1 and self.true_count%20 == 0):
                print "ROLLOUT DONE "
                self.r_count += 1
                self.grasp_count = 0
                self.success_count = 0
                self.grasp = True

            self.true_count += 1


    def save_image(self,c_img):
        if self.side == "BOTTOM":
            if self.grasp:
                cv2.imwrite(cfg.FAST_PATH + 'b_grasp/frame_'+str(self.r_count)+'_'+str(self.grasp_count)+'.png',c_img)
            else:
                cv2.imwrite(cfg.FAST_PATH + 'b_success/frame_'+str(self.r_count)+'_'+str(self.success_count)+'.png',c_img)

        else:
            if self.grasp:
                cv2.imwrite(cfg.FAST_PATH + 't_grasp/frame_'+str(self.r_count)+'_'+str(self.grasp_count)+'.png',c_img)
            else:
                cv2.imwrite(cfg.FAST_PATH + 't_success/frame_'+str(self.r_count)+'_'+str(self.success_count)+'.png',c_img)


if __name__ == "__main__":
    cp = BedMaker()
    cp.collect_data_bed()
