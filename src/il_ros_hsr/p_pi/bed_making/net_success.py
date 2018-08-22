import os, Pyro4, time, IPython, cv2, hsrb_interface, time
import cPickle as pickle
import numpy as np
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.sensors import  RGBD
from fast_grasp_detect.detectors.tran_detector import SDetector


class Success_Net:
    """The success network policy, check success by calling network.
    Actual net is in `fast_grasp_detect.detectors.tran_detector`.
    """

    def __init__(self, whole_body, tt, cam, base, fg_cfg, bed_cfg):
        self.whole_body = whole_body
        self.tt = tt
        self.cam = cam
        self.omni_base = base
        self.sdect = SDetector(fg_cfg=fg_cfg, bed_cfg=bed_cfg)


    def check_bottom_success(self,wl):
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'lower_start')
        #self.whole_body.move_to_joint_positions({'head_pan_joint': 1.5})
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        stranst = time.time()
        query_res = self.query_net(wl) # calls network
        etranst = time.time()
        print("Success predict time: " + str(etranst-stranst))
        return query_res


    def check_top_success(self,wl):
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'top_mid')
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        stranst = time.time()
        query_res = self.query_net(wl) # calls network
        etranst = time.time()
        print("Success predict time: " + str(etranst-stranst))
        return query_res


    def query_net(self,wl):
        """
        Calls success network, returns 3-element tuple: (boolean for success,
        the output from predicting, and the camera image that was input).
        """
        img = self.cam.read_color_data()
        sup_data = None
        time.sleep(4)
        img = self.cam.read_color_data()
        data = self.sdect.predict(np.copy(img))
        print "NET OUTPUT ",data
        ans = np.argmax(data)
        if (ans == 0):
            return True, data, img
        else:
            return False, data, img


if __name__ == "__main__":
    robot = hsrb_interface.Robot()
    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')
    cam = RGBD()
    wl = Python_Labeler(cam)
    sc = Success_Check(whole_body,None,cam,omni_base)
    print "RESULT ", sc.check_success(wl)
