import tf, IPython, os, sys, cv2, time, thread, rospy, glob, hsrb_interface
from tf import TransformListener
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped, Twist
from fast_grasp_detect.labelers.online_labeler import QueryLabeler
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from il_ros_hsr.core.sensors import RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick_X import JoyStick_X # Different JoyStick?
from il_ros_hsr.core.grasp_planner import GraspPlanner
from il_ros_hsr.core.rgbd_to_map import RGBD2Map
from il_ros_hsr.core.web_labeler import Web_Labeler
from il_ros_hsr.core.python_labeler import Python_Labeler
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.p_pi.bed_making.check_success import Success_Check
from il_ros_hsr.p_pi.bed_making.get_success import get_success
from il_ros_hsr.p_pi.bed_making.initial_state_sampler import InitialSampler
import numpy as np
import numpy.linalg as LA
from numpy.random import normal
from os.path import join
FAST_PATH = 'fast_path/'


class BedMaker():

    def __init__(self):
        """
        For faster data collection where we manually simulate it.
        We move with our hands.  This will give us the large datasets we need.

        NOTE ABOUT JOYSTICK: 
        """
        self.robot = robot = hsrb_interface.Robot()
        self.rgbd_map = RGBD2Map()
        self.omni_base = self.robot.get('omni_base')
        self.whole_body = self.robot.get('whole_body')
        self.cam = RGBD()
        self.com = COM()
        self.wl = Python_Labeler(cam=self.cam)

        # PARAMETERS TO CHANGE 
        # We choose a fixed side and collect data from there, no switching.
        # Automatically saves based on `r_count` and counting the saved files.
        # TODO grasp vs success?
        self.side = 'BOTTOM'
        self.grasp = True
        self.grasp_count = 0
        self.success_count = 0
        self.true_count = 0
        self.r_count = self.get_rollout_number()
        print("rollout number: {}".format(self.r_count))
        self.joystick = JoyStick_X(self.com)

        # Set up initial state, table, etc.
        self.com.go_to_initial_state(self.whole_body)
        self.tt = TableTop()

        # For now, a workaround. Ugly but it should do the job ...
        #self.tt.find_table(robot)
        self.tt.make_fake_ar()
        self.tt.find_table_workaround(robot)

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        time.sleep(4)

        # When we start, spin this so we can check the frames. Then un-comment,
        # etc. It's the current hack we have to get around crummy AR marker detection.
        #rospy.spin()

        # THEN we position the head since that involves moving the _base_.
        self.position_head()


    def get_rollout_number(self):
        """Had to modify this a bit from Michael's code. Test+see if it works.
        """
        if self.side == "BOTTOM":
            rollouts = glob.glob(FAST_PATH+'b_grasp/*.png')
        else:
            rollouts = glob.glob(FAST_PATH+'t_grasp/*.png')
        return len(rollouts)


    def position_head(self):
        """Ah, I see, we can go straight to the top. Whew."""

        # Original code:
        #if self.side == "TOP":
        #    self.tt.move_to_pose(self.omni_base,'right_down')
        #    self.tt.move_to_pose(self.omni_base,'right_up')
        #    self.tt.move_to_pose(self.omni_base,'top_mid')
        #    self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        #elif self.side == "BOTTOM":
        #    self.tt.move_to_pose(self.omni_base,'lower_start')
        #    self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})

        # New code reflecting the different poses and HSR joints:
        self.whole_body.move_to_go()
        if self.side == "BOTTOM":
            self.tt.move_to_pose(self.omni_base,'lower_start_tmp')
        else:
            self.tt.move_to_pose(self.omni_base,'right_down')
            self.tt.move_to_pose(self.omni_base,'right_up')
            self.tt.move_to_pose(self.omni_base,'top_mid_tmp')
        self.whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
        self.whole_body.move_to_joint_positions({'head_pan_joint':  np.pi/2.0})
        self.whole_body.move_to_joint_positions({'arm_lift_joint':  0.120})
        self.whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})


    def collect_data_bed(self):
        """Collect data using Michael Laskey's faster way.
        """
        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            # Continually show the camera image on screen and wait for user.
            # Doing `k=cv2.waitKey(33)` and `print(k)` results in -1 except for
            # when we press the joystick in a certain configuration.
            cv2.imshow('video_feed', c_img)
            cv2.waitKey(30)

            # Here's what they mean. Y is top, and going counterclockwise:
            # Y: [ 1,  0]
            # X: [-1,  0] # this is what we want for image collection
            # A: [ 0,  1]
            # B: [ 0, -1] # use to terminate the rollout
            #
            # There is also a bit of a delay embedded, i.e., repeated clicking
            # of `X` won't save images until some time has passed. Good! It is
            # also necessary to press for a few milliseconds (and not just tap).
            cur_recording = self.joystick.get_record_actions_passive()

            if (cur_recording[0] < -0.1 and self.true_count%20 == 0):
                print("PHOTO SNAPPED (cur_recording: {})".format(cur_recording))
                self.save_image(c_img, d_img)
                if self.grasp:
                    self.grasp_count += 1
                    self.grasp = False
                    print("self.grasp_count: {}".format(self.grasp_count))
                else:
                    self.success_count += 1
                    self.grasp = True
                    print("self.success_count: {}".format(self.grasp_count))

            # I would probably kill the script here to re-position and get more
            # diversity in the camera view that we have.
            if (cur_recording[1] < -0.1 and self.true_count%20 == 0):
                print("ROLLOUT DONE (cur_recording: {})".format(cur_recording))
                self.r_count += 1
                self.grasp_count = 0
                self.success_count = 0
                self.grasp = True
            self.true_count += 1


    def collect_data_grasp_only(self):
        """Collect data for the grasping network only, like H's method.
        """
        # TODO TODO TODO
        #while True:
        #    c_img = self.cam.read_color_data()
        #    d_img = self.cam.read_depth_data()
        pass


    def save_image(self, c_img, d_img):
        """Save images. Don't forget to process depth images.

        For now I'm using a tuned cutoff like 1400, at least to _visualize_.
        NOTE: since the cutoff for turning depth images into black may change,
        it would be better to save the original d_img in a dictionary. Don't use
        cv2.imwrite() as I know from experience that it won't work as desired.
        """
        f_rc_grasp   = 'frame_{}_{}.png'.format(self.r_count, self.grasp_count)
        f_rc_success = 'frame_{}_{}.png'.format(self.r_count, self.success_count)
        d_img = depth_to_net_dim(d_img, cutoff=1400)

        if self.side == "BOTTOM":
            if self.grasp:
                pth1 = join(FAST_PATH, 'b_grasp', 'rgb_'+f_rc_grasp)
                pth2 = join(FAST_PATH, 'b_grasp', 'depth_'+f_rc_grasp)
            else:
                pth1 = join(FAST_PATH, 'b_success', 'rgb_'+f_rc_success)
                pth2 = join(FAST_PATH, 'b_success', 'depth_'+f_rc_success)
        else:
            if self.grasp:
                pth1 = join(FAST_PATH, 't_grasp', 'rgb_'+f_rc_grasp)
                pth2 = join(FAST_PATH, 't_grasp', 'depth_'+f_rc_grasp)
            else:
                pth1 = join(FAST_PATH, 't_success', 'rgb_'+f_rc_success)
                pth2 = join(FAST_PATH, 't_success', 'depth_'+f_rc_success)
        cv2.imwrite(pth1, c_img)
        cv2.imwrite(pth2, d_img)


if __name__ == "__main__":
    cp = BedMaker()
    cp.collect_data_bed()
    #cp.collect_data_grasp_only()

