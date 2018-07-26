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
import cv2, time, thread, rospy
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
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
from online_labeler import QueryLabeler
from image_geometry import PinholeCameraModel as PCM
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from il_ros_hsr.core.rgbd_to_map import RGBD2Map
from il_ros_hsr.p_pi.bed_making.initial_state_sampler import InitialSampler

# Grasping and success.
from fast_grasp_detect.detectors.grasp_detector import GDetector
from il_ros_hsr.p_pi.bed_making.net_success import Success_Net


class BedMaker():

    def __init__(self):
        """For deploying the bed-making policy, not for data collection.
        Uses the neural network, `GDetector`, not the analytic baseline.
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
            self.wl = Python_Labeler(self.cam)

        # Set up initial state, table, etc.
        self.com.go_to_initial_state(self.whole_body)
        self.tt = TableTop()
        self.tt.find_table(self.robot)
        self.ins = InitialSampler(self.cam)
        self.side = 'BOTTOM'
        self.grasp_count = 0

        # Policy for grasp detection, using Deep Imitation Learning.
        self.g_detector = GDetector(cfg.GRASP_NET_NAME)
        self.sn = Success_Net(self.whole_body, self.tt, self.cam, self.omni_base)

        # Bells and whistles.
        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        self.gp = GraspPlanner()
        self.gripper = Bed_Gripper(self.gp,self.cam,self.com.Options,self.robot.get('gripper'))

        c_img = self.cam.read_color_data()
        self.sn.sdect.predict(c_img) # ???
        time.sleep(4)
        #thread.start_new_thread(self.ql.run,())
        print("after thread")


    def bed_make(self):
        """Runs the pipeline for deployment, testing out bed-making.
        You can run this for multiple bed-making trajectories.
        """
        self.rollout_stats = []
        self.get_new_grasp = True
        self.new_grasp = True

        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            if (not c_img == None and not d_img == None):
                if self.new_grasp:
                    self.position_head()
                else:
                    self.new_grasp = True
                time.sleep(3)
                c_img = self.cam.read_color_data()
                d_img = self.cam.read_depth_data()

                # Run grasp detector to get data=(x,y) point for grasp target.
                sgraspt = time.time()
                data = self.g_detector.predict(np.copy(c_img))
                egraspt = time.time()
                print("Grasp predict time: " + str(egraspt - sgraspt))
                self.record_stats(c_img,d_img,data,self.side,'grasp')

                # Broadcast grasp pose, execute the grasp, check for success.
                self.gripper.find_pick_region_net(data,c_img,d_img,self.grasp_count)
                pick_found,bed_pick = self.check_card_found()
                if self.side == "BOTTOM":
                    self.gripper.execute_grasp(bed_pick,self.whole_body,'head_down')
                else:
                    self.gripper.execute_grasp(bed_pick,self.whole_body,'head_up')
                self.check_success_state(c_img,d_img)


    def check_success_state(self,c_img,d_img):
        """
        Checks whether a single grasp in a bed-making trajectory succeeded.
        Depends on which side of the bed the HSR is at. Invokes the learned
        success network policy and transitions the HSR if successful.
        """
        if self.side == "BOTTOM":
            success, data, c_img = self.sn.check_bottom_success(self.wl)
        else:
            success, data, c_img = self.sn.check_top_success(self.wl)
        self.record_stats(c_img,d_img,data,self.side,'success')
        print "WAS SUCCESFUL: "
        print success

        # Handle transitioning to different side
        if success:
            if self.side == "BOTTOM":
                self.transition_to_top()
            else:
                self.transition_to_start()
            self.update_side()
        else:
            self.new_grasp = False
        self.grasp_count += 1

        # Limit amount of grasp attempts to cfg.GRASP_OUT (was 8 by default).
        if self.grasp_count > cfg.GRASP_OUT:
            self.transition_to_start()


    def update_side(self):
        """TODO: extend to multiple side switches?"""
        if self.side == "BOTTOM":
            self.side = "TOP"


    def transition_to_top(self):
        """Transition to top (not bottom)."""
        if cfg.DEBUG_MODE:
            self.com.save_stat(self.rollout_stats)
            self.tt.move_to_pose(self.omni_base,'lower_mid')
            sys.exit()
        else:
            self.move_to_top_side()


    def transition_to_start(self):
        """Transition to start=bottom, save rollout stats, exit program."""
        self.com.save_stat(self.rollout_stats)
        self.move_to_start()
        sys.exit()


    def record_stats(self,c_img,d_img,data,side,typ):
        """Adds a dictionary to the `rollout_stats` list."""
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
        """TODO we call the success network method ... this never called?"""
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


if __name__ == "__main__":
    cp = BedMaker()
    cp.bed_make()
