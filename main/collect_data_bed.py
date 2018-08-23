import tf, IPython, os, sys, cv2, time, thread, rospy, glob, hsrb_interface, argparse
from tf import TransformListener
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped, Twist
from fast_grasp_detect.labelers.online_labeler import QueryLabeler
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from il_ros_hsr.core.sensors import RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import JoyStick
from il_ros_hsr.core.grasp_planner import GraspPlanner
from il_ros_hsr.core.rgbd_to_map import RGBD2Map
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


class BedMaker():

    def __init__(self, args):
        """For data collection of bed-making, NOT the deployment.

        Assumes we roll out the robot's policy via code (not via human touch).
        This is the 'slower' way where we have the python interface that the
        human clicks on to indicate grasping points. Good news is, our deployment
        code is probably going to be similar to this.

        For joystick: you only need it plugged in for the initial state sampler,
        which (at the moment) we are not even using.
        """
        self.robot = robot = hsrb_interface.Robot()
        self.rgbd_map = RGBD2Map()
        self.omni_base = robot.get('omni_base')
        self.whole_body = robot.get('whole_body')
        self.cam = RGBD()
        self.com = COM()
        self.wl = Python_Labeler(cam=self.cam)

        # View mode: STANDARD (the way I was doing earlier), CLOSE (the way they want).
        self.view_mode = cfg.VIEW_MODE

        # Set up initial state, table, etc.
        self.com.go_to_initial_state(self.whole_body)
        self.tt = TableTop()

        # For now, a workaround. Ugly but it should do the job ...
        #self.tt.find_table(robot)
        self.tt.make_fake_ar()
        self.tt.find_table_workaround(robot)

        #self.ins = InitialSampler(self.cam)
        self.side = 'BOTTOM'
        self.grasp_count = 0

        # Bells and whistles; note the 'success check' to check if transitioning
        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()
        self.gp = GraspPlanner()
        self.gripper = Bed_Gripper(self.gp, self.cam, self.com.Options, robot.get('gripper'))
        self.sc = Success_Check(self.whole_body, self.tt, self.cam, self.omni_base)

        time.sleep(4)
        print("Finished creating BedMaker()! Get the bed set up and run bed-making!")
        if cfg.INS_SAMPLE:
            print("TODO: we don't have sampling code here.")

        # When we start, spin this so we can check the frames. Then un-comment,
        # etc. It's the current hack we have to get around crummy AR marker detection.
        if args.phase == 1:
            print("Now doing rospy.spin() because phase = 1.")
            rospy.spin()


    def bed_make(self):
        """Runs the pipeline for data collection.

        You can run this for multiple bed-making trajectories.
        For now, though, assume one call to this means one trajectory.
        """
        self.rollout_data = []
        self.get_new_grasp = True

        # I think, creates red line in GUI where we adjust the bed to match it.
        # But in general we better fix our sampler before doing this for real.
        # Don't forget to press 'B' on the joystick to get past this screen.
        if cfg.INS_SAMPLE:
            u_c, d_c = self.ins.sample_initial_state()
            self.rollout_data.append( [u_c, d_c] )

        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            if (not c_img.all() == None and not d_img.all() == None):
                if self.get_new_grasp:
                    self.position_head()

                    # Human supervisor labels. data = dictionary of relevant info
                    data = self.wl.label_image(c_img)
                    c_img = self.cam.read_color_data()
                    d_img = self.cam.read_depth_data()
                    self.add_data_point(c_img, d_img, data, self.side, 'grasp')

                    # Broadcasts grasp pose
                    self.gripper.find_pick_region_labeler(data, c_img, d_img, self.grasp_count)

                # Execute the grasp and check for success. But if VIEW_MODE is
                # close, better to reset to a 'nicer' position for base movement.
                pick_found, bed_pick = self.check_card_found()
                if self.side == "BOTTOM":
                    self.whole_body.move_to_go()
                    self.tt.move_to_pose(self.omni_base,'lower_start')
                    self.gripper.execute_grasp(bed_pick, self.whole_body, 'head_down')
                else:
                    self.whole_body.move_to_go()
                    self.tt.move_to_pose(self.omni_base,'top_mid')
                    self.gripper.execute_grasp(bed_pick, self.whole_body, 'head_up')
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
        self.add_data_point(c_img, d_img, data, self.side, 'success')
        print("WAS SUCCESFUL: {}".format(success))

        # Handle transitioning to different side
        if success:
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


    def update_side(self):
        """TODO: extend to multiple side switches?"""
        if self.side == "BOTTOM":
            self.side = "TOP"


    def transition_to_top(self):
        """Transition to top (not bottom)."""
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
        """Position the head for a grasp attempt.
        After playing around a bit, I think `head_tilt_joint` should be set last.
        """
        self.whole_body.move_to_go()
        if self.side == "BOTTOM":
            self.tt.move_to_pose(self.omni_base,'lower_start_tmp')
        self.whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
        self.whole_body.move_to_joint_positions({'head_pan_joint':  np.pi/2.0})
        self.whole_body.move_to_joint_positions({'arm_lift_joint':  0.120})
        self.whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})


    def move_to_top_side(self):
        """Assumes we're at the bottom and want to go to the top."""
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'right_down')
        self.tt.move_to_pose(self.omni_base,'right_mid')
        self.tt.move_to_pose(self.omni_base,'right_up')
        self.tt.move_to_pose(self.omni_base,'top_mid_tmp')


    def move_to_start(self):
        """Assumes we're at the top and we go back to the start."""
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'right_up')
        self.tt.move_to_pose(self.omni_base,'right_mid')
        self.tt.move_to_pose(self.omni_base,'right_down')
        self.tt.move_to_pose(self.omni_base,'lower_mid')
        

    def check_card_found(self):
        """Looks up the pose for where the HSR's hand should go to."""
        transforms = self.tl.getFrameStrings()
        cards = []
        try:
            for transform in transforms:
                current_grasp = 'bed_'+str(self.grasp_count)
                if current_grasp in transform:
                    print('found {}'.format(current_grasp))
                    f_p = self.tl.lookupTransform('map',transform, rospy.Time(0))
                    cards.append(transform)
        except: 
            rospy.logerr('bed pick not found yet')
        return True, cards


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--phase', type=int, help='1 for checking poses, 2 for deployment.')
    args = pp.parse_args()
    assert args.phase in [1,2]

    cp = BedMaker(args)
    cp.bed_make()
    rospy.spin()
