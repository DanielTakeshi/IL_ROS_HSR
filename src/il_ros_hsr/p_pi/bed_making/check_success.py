import os, Pyro4, time, IPython, cv2, hsrb_interface
import cPickle as pickle
import numpy as np
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler
from il_ros_hsr.core.sensors import  RGBD
CANVAS_DIM = 420.0


class Success_Check:
    """Checking for success during the data collection.
    
    See `net_success.py` for the DNN version.
    """

    def __init__(self,whole_body,tt,cam,base):
        self.cam = cam
        self.whole_body = whole_body
        self.tt = tt
        self.omni_base = base


    def check_bottom_success(self,wl):
        """I think it's safer to set `whole_body.move_to_go()` first."""
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'lower_start_tmp')
        self.whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
        self.whole_body.move_to_joint_positions({'head_pan_joint':  np.pi/2.0})
        self.whole_body.move_to_joint_positions({'arm_lift_joint':  0.120})
        self.whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})# -np.pi/36.0})
        return self.check_success(wl)


    def check_top_success(self,wl):
        """I think it's safer to set `whole_body.move_to_go()` first."""
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'top_mid_tmp')
        self.whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
        self.whole_body.move_to_joint_positions({'head_pan_joint':  np.pi/2.0})
        self.whole_body.move_to_joint_positions({'arm_lift_joint':  0.120})
        self.whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})# -np.pi/36.0})
        return self.check_success(wl)


    def check_success(self,wl):
        """Check for success by querying supervisor via `wl.label_image(img)`. 
        
        The data we get from the web labeler consists of a dictionary, with the
        'objects' key being a _list_ of dictionaries, hence the iteration over
        that. See `QueryLabeler.setClass()` in the `fast_grasp_detect` library
        for the classes. If the class is zero, then it is SUCCESS and we should
        transition to the other side. Else, failure and re-try grasping.
        """
        img = self.cam.read_color_data()
        data = wl.label_image(img)

        print("\nInside bed_making/check_success.py, the check_success()` method")
        print("our data from `wl.label_image(img)`:")
        for key in data:
            print("data[{}]: {}".format(key, data[key]))

        for result in data['objects']:
            # See `QueryLabeler` (e.g., `setClass` method) if confused.
            if (result['class'] == 0): 
                return True, data
            else:
                return False, data


if __name__ == "__main__":
    robot = hsrb_interface.Robot()
    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')
    cam = RGBD()
    wl = Python_Labeler(cam)
    sc = Success_Check(whole_body,None,cam,omni_base)
    print "RESULT ", sc.check_success(wl)
