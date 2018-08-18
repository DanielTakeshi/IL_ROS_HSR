import sys, cv2, time, IPython, tf, rospy, thread, hsrb_interface, geometry_msgs
from tf import TransformListener, TransformBroadcaster
import controller_manager_msgs.srv
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped, Point, WrenchStamped
from hsrb_interface import geometry
from hsrb_interface.collision_world import CollisionWorld
from cv_bridge import CvBridge, CvBridgeError
from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick
from il_ros_hsr.core.rgbd_to_map import RGBD2Map
from il_ros_hsr.core.grasp_planner import GraspPlanner
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.tensioner import Tensioner
from il_ros_hsr.core.python_labeler import Python_Labeler
from il_ros_hsr.core.web_labeler import Web_Labeler
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from image_geometry import PinholeCameraModel as PCM
from numpy.random import normal
import numpy as np
import numpy.linalg as LA


TABLE_HEIGHT = 0.61
TABLE_WIDTH = 0.67
TABLE_OFFSET = 0.06
OFFSET = 0.5
OFFSET_T = 0.34
TABLE_LENGTH = 0.91


class TableTop():

    def __init__(self):
        self.tl = TransformListener()
        self.br = TransformBroadcaster()

    def loop_broadcast(self, pos, rot, name, ref):
        while True:
            self.br.sendTransform(pos, rot, rospy.Time.now(), name, ref)

    def simulate_ar(self):
        # A hand-tuned y-offset value. If we follow blue tape, AR should be OK.
        pos = (0.0, 0.58, 0.0)
        rot = tf.transformations.quaternion_from_euler(ai=0.0, aj=np.pi, ak=0.0)
        name = 'T_ar'
        ref = 'map'
        thread.start_new_thread(self.loop_broadcast, (pos, rot, name, ref))

    def simulate_head_down(self):
        # Pretend we know the head down as well, where we have a target.
        pos = np.array([-(TABLE_LENGTH/2.0+0.02), OFFSET_T+0.04, -TABLE_HEIGHT+0.04])
        rot = tf.transformations.quaternion_from_euler(ai=0.0, aj=0.0, ak=np.pi/2.0)
        name = 'T_head_down'
        ref = 'T_ar'
        thread.start_new_thread(self.loop_broadcast, (pos, rot, name, ref))

    def simulate_grasp(self, count):
        # Pretend we know the grasp point.
        pos = np.array([0.0, OFFSET_T+0.04+0.20, -TABLE_HEIGHT+0.04])
        rot = tf.transformations.quaternion_from_euler(ai=0.0, aj=0.0, ak=np.pi/2.0)
        name = 'T_grasp_'+str(count)
        ref = 'T_ar'
        thread.start_new_thread(self.loop_broadcast, (pos, rot, name, ref))

    def check_card_found(self, count):
        """Looks up the pose for where the HSR's hand should go to."""
        print("\nbegin check_card_found\n")
        transforms = self.tl.getFrameStrings()
        cards = []
        try:
            for transform in transforms:
                print(transform)
                current_grasp = 'T_grasp_'+str(count)
                if current_grasp in transform:
                    print('\tgot here')
                    #f_p = self.tl.lookupTransform('map', transform, rospy.Time(0))
                    f_p = self.tl.lookupTransform('base_link', transform, rospy.Time(0))
                    cards.append(transform)
        except: 
            rospy.logerr('bed pick not found yet')
        print("\nend check_card_found\n")
        return True, cards

    def execute_grasp(self, cards, whole_body, direction, com, gripper, tension):
        """ Executes grasp. Move to pose, squeeze, pull (w/tension), open. """
        print("executing grasps:\n\t{}, {}\n".format(cards[0], direction))
        whole_body.end_effector_frame = 'hand_palm_link'
        whole_body.move_end_effector_pose(geometry.pose(), cards[0])
        com.grip_squeeze(gripper)

        # Can't do this, must do something else.
        #tension.force_pull(whole_body,direction)
        # Unfortunately this results in the same thing! It goes out of whack,
        # different direction (almost to the exact opposite of where it should
        # be going for some reason ...).

        #whole_body.move_end_effector_pose(geometry.pose(), direction)
        # We want to move in the _positive_ y direction.
        #whole_body.move_end_effector_pose(geometry.pose(y=0.20), cards[0])
        whole_body.move_end_effector_pose(geometry.pose(), direction)

        # Back to normal, open gripper
        com.grip_open(gripper)



if __name__ == "__main__":
    robot = hsrb_interface.Robot()
    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')
    com = COM()
    cam = RGBD()
    gp = GraspPlanner()
    gripper = Bed_Gripper(gp, cam, com.Options, robot.get('gripper'))
    tension = Tensioner()

    if cfg.USE_WEB_INTERFACE:
        wl = Web_Labeler()
    else:
        wl = Python_Labeler(cam=cam)

    # Should be same as `com.go_to_initial_position()` but we may want some extra stuff.
    whole_body.move_to_go()
    whole_body.move_to_joint_positions({'head_pan_joint': 1.5})
    whole_body.move_to_joint_positions({'head_tilt_joint': -0.8})
    #whole_body.move_to_joint_positions({'head_tilt_joint': ...}) # make it taller?

    print("We've now gone to the initial state. Let's pause a few seconds and then find the table...")
    time.sleep(3)

    # Fake bed.
    tt = TableTop()
    tt.simulate_ar() # Make the AR but have this be wrt map frame.
    time.sleep(1)
    tt.simulate_head_down() # Should be target location of where to grasp on the bed.
    time.sleep(1)
    tt.simulate_grasp(count=1)
    time.sleep(1)
    pick_found, bed_pick = tt.check_card_found(count=1)
    result = bed_pick[0] # a string representing the topic name, `T_grasp_1`
    print("The transform: {}.\n(Now let's execute the grasp...)\n".format(result))

    # Don't do this until we know we've checked the grasp pose!
    do_it = False

    if do_it:
        # Normally I'd do:
        #pick_found,bed_pick = self.check_card_found()
        #self.gripper.execute_grasp(bed_pick,self.whole_body,'head_down')
        # where:
        # self.gripper = Bed_Gripper(self.gp,self.cam,self.com.Options,self.robot.get('gripper'))

        # But here we do:
        #gripper.execute_grasp(bed_pick, whole_body, 'T_head_down')
        # UPDATE: no the direction is screwed up somehow ... let's just do things here:
        # also note, gripper = Bed_Gripper(...) has its own gripper argument from robot.get('gripper')

        tt.execute_grasp(bed_pick, whole_body, 'T_head_down', com, gripper.gripper, tension)

        # Move to go (i.e., turn into 'default' stance) while also moving back
        # to the starting map. Michael used a different frame (not 'map') which
        # is a better idea anyway.
        whole_body.move_to_go()
        omni_base.move(geometry.pose(), 500.0, ref_frame_id='map')
        whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        whole_body.move_to_joint_positions({'head_pan_joint': 1.5})

    rospy.spin()
