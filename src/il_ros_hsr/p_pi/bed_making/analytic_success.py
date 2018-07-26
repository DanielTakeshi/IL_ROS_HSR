import os
import Pyro4
import time
import cPickle as pickle
import IPython
import cv2
import numpy as np
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler
import hsrb_interface
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from il_ros_hsr.core.sensors import  RGBD
from il_ros_hsr.p_pi.bed_making.analytic_transition import Analytic_Transition
import time
GLOBAL_PATH = "/home/autolab/Workspaces/michael_working/IL_ROS_HSR/"


class Success_Net:
    """The success network with analytic policy, but I thought we didn't use...
    """

    def __init__(self,whole_body,tt,cam,base):
        self.cam = cam
        self.whole_body = whole_body
        self.tt = tt
        self.omni_base = base
        self.sdect = Analytic_Transition()


    def check_bottom_success(self,wl):
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'lower_start')
        #self.whole_body.move_to_joint_positions({'head_pan_joint': 1.5})
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        stranst = time.time()
        query_res = self.query_net(wl)
        etranst = time.time()
        print("Success predict time: " + str(etranst-stranst))
        return query_res


    def check_top_success(self,wl):
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'top_mid')
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        stranst = time.time()
        query_res = self.query_net(wl)
        etranst = time.time()
        print("Success predict time: " + str(etranst-stranst))
        return query_res


    def query_net(self,wl):
        img = self.cam.read_color_data()
        sup_data = None
        time.sleep(4)
        img = self.cam.read_color_data()
        # CHANGE HERE (unlike with analytic grasp, no need to do 3*ans).
        predict_scale = 3
        img_small = cv2.resize(np.copy(img),(640/3,480/3))
        ans = self.sdect.predict(img_small, predict_scale)
        return ans,sup_data,img


if __name__ == "__main__":
    robot = hsrb_interface.Robot()
    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')
    cam = RGBD()
    wl = Python_Labeler(cam)
    sc = Success_Check(whole_body,None,cam,omni_base)
    print "RESULT ", sc.check_success(wl)
