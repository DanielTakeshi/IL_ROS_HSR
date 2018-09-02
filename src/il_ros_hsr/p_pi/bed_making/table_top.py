import sys, cv2, time, IPython, tf, rospy, thread, hsrb_interface, geometry_msgs
from geometry_msgs.msg import PoseStamped, Point, Quaternion, WrenchStamped, Pose, Twist, TransformStamped
from hsrb_interface import geometry
from image_geometry import PinholeCameraModel as PCM
from cv_bridge import CvBridge, CvBridgeError
from il_ros_hsr.core.sensors import RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import JoyStick
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from tf import TransformListener, TransformBroadcaster
import numpy as np
import numpy.linalg as LA

# OFFSETs. To decide these, set other offsets in the transformations code to 0,
# then adjust these below until the coordinate frames for the bed match the
# corners. Then add more offsets in the code to make the corners 'further
# outwards' since the corners are actually grasing points.

# Long side of top bed frame is about 91 cm, x-axis wrt AR marker.
TABLE_LENGTH = 0.91

# Shorter side to bed frame is about 67 cm, y-axis wrt AR marker.
TABLE_WIDTH = 0.67

# z-axis wrt AR, but actual table height is about 46cm, but I think it's because
# we want the HSR's target grip pose to be about 15cm above the actual bed.
# Though, confusingly, we add 0.04 to the negative of this so in practice we set
# the height of the head up/down poses to be about 57cm, so ~11cm taller.
TABLE_HEIGHT = 0.61

# For the bottom up/down, but we don't use these at all so probably ignore...
TABLE_OFFSET = 0.06

# Offset in y-axis representing distance from AR marker to the closer edge of
# the bed. So, e.g., all four of the bed corner frames need this added to them.
# Before, 0.34. For us I think it's 0.36 now ...
OFFSET_T = 0.36

# Used for multiple offsets regarding the _trajectory_ the HSR takes.
OFFSET = 0.5


class TableTop():

    def __init__(self):
        self.tl = TransformListener()
        self.br = TransformBroadcaster()

        # For our ugly workaorund. Please assume the AR is only offset from the
        # map by (a) the y-coordinate, and (b) the aj angle. Otherwise we'll
        # have to fix a lot of things downstream in the code ...
        self.FAKE_ai = 0.0
        self.FAKE_aj = np.pi
        self.FAKE_ak = 0.0
        self.FAKE_AR_ROT = tf.transformations.quaternion_from_euler(
                ai=self.FAKE_ai, aj=self.FAKE_aj, ak=self.FAKE_ak)
        self.FAKE_AR_POS = (0.0, 0.58, 0.0)
   

    def broadcast_pose(self,pose,label):
        while True:
            self.br.sendTransform(pose['trans'], pose['quat'], rospy.Time.now(), label, 'map')
    

    def move_to_pose(self,base,label):
        base.move(geometry.pose(), 500.0, ref_frame_id=label)


    def cal_transform(self, offsets, rot=None):
        """
        Make translation (L_t_trans) and rotation (L_t_rot) matrices from
        the offsets and rotations we created wrt the AR MARKER pose. However, we
        have to then map that back to the map frame.

        This will give us a final RBT, which we return separately.
        """
        L_t_trans = tf.transformations.translation_matrix(offsets)
        M_t_L = np.matmul(self.M_t_A, L_t_trans)

        if not (rot is None):
            q_rot = tf.transformations.quaternion_from_euler(ai=rot[0],aj=rot[1],ak=rot[2])
            L_t_rot = tf.transformations.quaternion_matrix(q_rot)
            L_t_rot[:,3] = L_t_trans[:,3]
            M_t_L = np.matmul(self.M_t_A, L_t_rot)

        trans = tf.transformations.translation_from_matrix(M_t_L)
        quat = tf.transformations.quaternion_from_matrix(M_t_L)
        return trans, quat


    def make_new_pose(self, offsets, label, rot=None):
        t,q = self.cal_transform(offsets, rot=rot)
        pose = {}
        pose['trans'] = t
        pose['quat'] = q
        thread.start_new_thread(self.broadcast_pose, (pose,label))


    def calculat_ar_in_map(self, obj):
        """Gets AR and map frames set up.
        
        Make the frames wrt the AR marker, but need a transformation involving
        `map` because I think the latter will stay fixed throughout but the AR
        marker will move as the two stereo cameras move.

        Update: no, pretty sure the AR marker should stay fixed? I think this
        was my code where I was pretending to find the AR marker...
        """
        ar_pose = obj.get_pose(ref_frame_id = 'ar_marker/11')
        marker_pose = obj.get_pose(ref_frame_id='map')
        
        M_t_O = tf.transformations.quaternion_matrix(marker_pose.ori)
        M_t_trans = tf.transformations.translation_matrix(marker_pose.pos)
        M_t_O[:,3] = M_t_trans[:,3]

        A_t_O = tf.transformations.quaternion_matrix(ar_pose.ori)
        A_t_trans = tf.transformations.translation_matrix(ar_pose.pos)
        A_t_O[:,3] = A_t_trans[:,3]

        self.M_t_A = np.matmul(M_t_O,LA.inv(A_t_O))
        trans = tf.transformations.translation_from_matrix(self.M_t_A)
        quat = tf.transformations.quaternion_from_matrix(self.M_t_A)
        print("calculat_ar_in_map: TRANS {}".format(trans))
        print("calculat_ar_in_map: ROTATION {}".format(quat))
        return trans,quat

   
    ##def find_table(self, robot):
    ##    """Creates the various poses that we need for bed-making.

    ##    Unfortunately, the robot detector seems to be malfunctioning. Here's the docs:
    ##    https://docs.hsr.io/archives/manual/1710/en/reference/python.html#objectdetector

    ##    Angles are in radians, so 1.57 is approx pi/2 or 90 degrees.

    ##    Must be wrt AR marker, so e.g., a negative x-axis means moving closer to
    ##    the HEAD up/down sides of the bed, a negative y-axis offset means moving
    ##    AWAY from the bed, and a negative z-axis means moving UPWARDS.
    ##    """

    ##    # ---- This stuff is not working ----
    ##    detector = robot.get("marker")
    ##    sd = detector.get_objects()
    ##    print("detector: {}".format(detector))
    ##    print("detector.get_objects(): {}".format(sd))
    ##    trans, quat = self.calculat_ar_in_map(obj=sd[0])

    ##    # ---- The four corners of the bed, or more accurately, grasp targets. ----

    ##    # HEAD DOWN, corner for first (bottom) side of bed, but has offset for grasp target.
    ##    offsets = np.array([-(TABLE_LENGTH/2.0+0.02), OFFSET_T+0.04, -TABLE_HEIGHT+0.04])
    ##    rot = np.array([0.0,0.0,1.57])
    ##    self.make_new_pose(offsets,'head_down',rot = rot)

    ##    # HEAD UP, corner for second (top) side of the bed, but has offset for grasp target.
    ##    # As expected, only difference with HEAD UP is that we add TABLE_WIDTH to the y-axis.
    ##    offsets = np.array([-(TABLE_LENGTH/2.0+0.02), (OFFSET_T+TABLE_WIDTH+0.02), -TABLE_HEIGHT+0.04])
    ##    rot = np.array([0.0,0.0,-1.57])
    ##    self.make_new_pose(offsets,'head_up',rot = rot)

    ##    # BOTTOM DOWN AT TABLE HEIGHT.
    ##    offsets = np.array([(TABLE_LENGTH/2.0+0.08), OFFSET_T+0.04, -TABLE_HEIGHT+TABLE_OFFSET])
    ##    rot = np.array([0.0,0.0,1.57])
    ##    self.make_new_pose(offsets,'bottom_down',rot = rot)

    ##    # BOTTOM UP AT TABLE HEIGHT
    ##    offsets = np.array([(TABLE_LENGTH/2.0+0.08), (OFFSET_T+TABLE_WIDTH+0.02), -TABLE_HEIGHT+TABLE_OFFSET])
    ##    rot = np.array([0.0,0.0,-1.57])
    ##    self.make_new_pose(offsets,'bottom_up',rot = rot)

    ##    # ---- The trajectory of the HSR. ----

    ##    # LOWER MID, where HSR begins, ideally we can see AR marker 11 from here.
    ##    offsets = np.array([0.0,-OFFSET-0.07,0.0])
    ##    rot = np.array([0.0,0.0,-3.14])
    ##    self.make_new_pose(offsets,'lower_mid',rot=rot)

    ##    # LOWER MID, HSR starts bed-making by moving here from `lower_mid`, so
    ##    # it moves closer to the bed, and then receives the image of the setup.
    ##    offsets = np.array([0.0,-OFFSET+0.16,0.0])
    ##    rot = np.array([0.0,0.0,1.57])
    ##    self.make_new_pose(offsets,'lower_start',rot=rot)

    ##    # RIGHT CORNER, go from `lower_start` to `right_down` after we finish grasp.
    ##    offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0), 0.0, 0.0])
    ##    self.make_new_pose(offsets,'right_down')

    ##    # RIGHT MID, go from `right_down` to `right_mid`. (This might get skipped, actually)
    ##    offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0),(OFFSET+TABLE_WIDTH/2.0),0.0])
    ##    self.make_new_pose(offsets,'right_mid')
 
    ##    # TOP CORNER, go from `right_mid` to `right_up`.
    ##    offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0), (2*OFFSET+TABLE_WIDTH), 0.0])
    ##    self.make_new_pose(offsets,'right_up')

    ##    # TOP MID, where HSR goes to see the bed from top side, and performs grasp.
    ##    offsets = np.array([0.0, (2*OFFSET+TABLE_WIDTH), 0.0])
    ##    rot = np.array([0.0,0.0,-1.57])
    ##    self.make_new_pose(offsets,'top_mid',rot=rot)

    ##    # Then after this, we go in reverse, to `right_up`, etc., all the way
    ##    # back to `lower_mid`, NOT `lower_start`!!!
    ##    #
    ##    # If we want the robot to have a closer-view of the bed, as with Honda's
    ##    # setup, we need `top_mid` to be closer to the bed.
  
    ##    # ---- Not sure when we use these? ----

    ##    # TOP MID FAR, like 'top_mid' except further away.
    ##    offsets = np.array([0.0, 3 * OFFSET + TABLE_WIDTH, 0.0])
    ##    rot = np.array([0.0,0.0,1.57])
    ##    self.make_new_pose(offsets,'top_mid_far',rot = rot)

    ##    # TOP LEFT FAR, like 'top_mid' except further away and also further to
    ##    # the left, so it's closer to the part the _human_ tucked-in, on other side.
    ##    offsets = np.array([(TABLE_WIDTH/2.0), 3 * OFFSET + TABLE_WIDTH, 0.0])
    ##    rot = np.array([0.0, 0.0, 3.14 * 1.0/4.0])
    ##    self.make_new_pose(offsets, 'top_left_far',rot = rot)


    # WORKAROUNDS

    def find_table_workaround(self, robot):
        """Creates the various poses that we need for bed-making.

        Unfortunately, the robot detector seems to be malfunctioning. Here's the docs:
        https://docs.hsr.io/archives/manual/1710/en/reference/python.html#objectdetector

        Angles are in radians, so 1.57 is approx pi/2 or 90 degrees.

        Must be wrt AR marker, so e.g., a negative x-axis means moving closer to
        the HEAD up/down sides of the bed, a negative y-axis offset means moving
        AWAY from the bed, and a negative z-axis means moving UPWARDS.
        """
        # ------------------------------------------------------------------------------------------
        # ---- The four corners of the bed, or more accurately, grasp targets. ----
        # ------------------------------------------------------------------------------------------
        # Note: head_up and head_down are what we actually use for grasp pulling targets.
        # ------------------------------------------------------------------------------------------

        # x-offset for head up/down. Increase this to make grasp target further away from corner, 
        # towards the dvrk machines. Probably a value like 0.04 or so will work ...
        HX_OFF = 0.05

        # If this is zero, then the four corners should have y-axis that are roughly coinciding with
        # the bed's boundaries. (It's tricky for the opposite side which we can't see easily.)
        # Increase this to make corners (grasp targets) slightly outside. Also applies to the bottom
        # frames. I think values of 0.06 or so will work... For cal sheet it is a bit wider.
        # ------------------------------------------------------------------------------------------
        HY_OFF = 0.07
        if cfg.BLANKET == 'cal':
            HY_OFF += 0.02

        # By default, the z-axis for the head_up & head_down poses will be 15cm ABOVE the actual
        # table height. Thus, add this offset to DECREASE the height (since z-axis is pointing
        # down).  E.g., using 0.04 (as we did earlier) will cause the heights to be 11cm. Also
        # applies to the bottom frames.
        # ------------------------------------------------------------------------------------------
        HZ_OFF = 0.04

        # HEAD DOWN, corner for first (bottom) side of bed, but has offset for grasp target.
        offsets = np.array([-(TABLE_LENGTH/2.0 + HX_OFF), OFFSET_T-HY_OFF, -TABLE_HEIGHT+HZ_OFF])
        rot = np.array([0.0,0.0,1.57])
        self.new_pose_workaround(offsets, 'head_down', rot)

        # HEAD UP, corner for second (top) side of the bed, but has offset for grasp target.
        # As expected, only difference with HEAD DOWN is that we add TABLE_WIDTH to the y-axis.
        # Well, no, we can add more offset so that the poses are not actually corners but a 
        # few cm 'outside' of one, so a successful grasp+pull will pull them to a good spot.
        # ------------------------------------------------------------------------------------------
        offsets = np.array([-(TABLE_LENGTH/2.0 + HX_OFF), (OFFSET_T+TABLE_WIDTH+HY_OFF), -TABLE_HEIGHT+HZ_OFF])
        rot = np.array([0.0,0.0,-1.57])
        self.new_pose_workaround(offsets, 'head_up', rot)

        # BOTTOM DOWN AT TABLE HEIGHT.
        offsets = np.array([(TABLE_LENGTH/2.0 + 0.08), OFFSET_T-HY_OFF, -TABLE_HEIGHT+HZ_OFF])
        rot = np.array([0.0,0.0,1.57])
        self.new_pose_workaround(offsets, 'bottom_down', rot)

        # BOTTOM UP AT TABLE HEIGHT
        offsets = np.array([(TABLE_LENGTH/2.0 + 0.08), (OFFSET_T+TABLE_WIDTH+HY_OFF), -TABLE_HEIGHT+HZ_OFF])
        rot = np.array([0.0,0.0,-1.57])
        self.new_pose_workaround(offsets, 'bottom_up', rot)

        # ------------------------------------------------------------------------------------------
        # ---- The trajectory of the HSR. ----
        # ------------------------------------------------------------------------------------------

        # LOWER MID, where HSR begins, ideally we can see AR marker 11 from here.
        offsets = np.array([0.0,-OFFSET-0.07,0.0])
        rot = np.array([0.0,0.0,-3.14])
        self.new_pose_workaround(offsets, 'lower_mid', rot)

        # LOWER MID, HSR starts bed-making by moving here from `lower_mid`, so
        # it moves closer to the bed, and then receives the image of the setup.
        # If using alternative view, we actually need to change the z-rotation.
        # But _after_ that, we'll go back to this normal pose to help w/base movement.

        if cfg.VIEW_MODE == 'close':
            offsets = np.array([-0.14, -0.14, 0.0])
            rot = np.array([0.0, 0.0, np.pi/2.0])
            self.new_pose_workaround(offsets, 'lower_start', rot)
            rot = np.array([0.0, 0.0, np.pi])
            self.new_pose_workaround(offsets, 'lower_start_tmp', rot)
        else:
            offsets = np.array([-0.05, -OFFSET+0.16 - 0.05, 0.0])
            rot = np.array([0.0, 0.0, np.pi/2.0])
            self.new_pose_workaround(offsets, 'lower_start', rot)
            rot = np.array([0.0, 0.0, np.pi])
            self.new_pose_workaround(offsets, 'lower_start_tmp', rot)

        # ------------------------------------------------------------------------------------------
        # For the corners, I am making two versions. This is so the HSR moves
        # 'straight' when we go to different sides, instead of moving sideways.
        # When moving to the top, use `right_X_1`, etc.
        # ------------------------------------------------------------------------------------------

        # RIGHT CORNER, go from `lower_start` to `right_down` after we finish grasp.
        offsets = np.array([-(OFFSET + TABLE_LENGTH/2.0 - 0.10), 0.0, 0.0])
        rot = np.array([0.0, 0.0, np.pi/2.0])
        self.new_pose_workaround(offsets, 'right_down_1', rot=rot)
        rot = np.array([0.0, 0.0, -np.pi/2.0])
        self.new_pose_workaround(offsets, 'right_down_2', rot=rot)

        # RIGHT MID, go from `right_down` to `right_mid`. (This may get skipped)
        offsets = np.array([-(OFFSET + TABLE_LENGTH/2.0 - 0.10), (OFFSET + TABLE_WIDTH/2.0), 0.0])
        rot = np.array([0.0, 0.0, np.pi/2.0])
        self.new_pose_workaround(offsets, 'right_mid_1', rot=rot)
        rot = np.array([0.0, 0.0, -np.pi/2.0])
        self.new_pose_workaround(offsets, 'right_mid_2', rot=rot)
 
        # TOP CORNER, go from `right_mid` to `right_up`. Update: added a few -10cm offsets.
        offsets = np.array([-(OFFSET + TABLE_LENGTH/2.0 - 0.10), (2*OFFSET+TABLE_WIDTH - 0.10), 0.0])
        rot = np.array([0.0, 0.0, np.pi/2.0])
        self.new_pose_workaround(offsets, 'right_up_1', rot=rot)
        rot = np.array([0.0, 0.0, -np.pi/2.0])
        self.new_pose_workaround(offsets, 'right_up_2', rot=rot)

        # TOP MID, where HSR goes to see the bed from top side, and performs grasp.
        if cfg.VIEW_MODE == 'close':
            # x-axis of -0.14 is to align w/bottom; y-axis unfortunately requires tuning.
            offsets = np.array([-0.14, (2*OFFSET + TABLE_WIDTH) - 0.10, 0.0])
            rot = np.array([0.0, 0.0, -np.pi/2.0])
            self.new_pose_workaround(offsets, 'top_mid', rot)
            rot = np.array([0.0, 0.0, 0.0])
            self.new_pose_workaround(offsets, 'top_mid_tmp', rot)
        else:
            # Offsets should ideally match the bottom but they don't ... yeah. :-(
            offsets = np.array([ 0.00, (2*OFFSET + TABLE_WIDTH) + 0.16, 0.0])
            rot = np.array([0.0, 0.0, -np.pi/2.0])
            self.new_pose_workaround(offsets, 'top_mid', rot)
            rot = np.array([0.0, 0.0, 0.0])
            self.new_pose_workaround(offsets, 'top_mid_tmp', rot)

        # ------------------------------------------------------------------------------------------
        # Then after this, we go in reverse, to `right_up`, etc., all the way
        # back to `lower_mid`, NOT `lower_start`!!!
        #
        # If we want the robot to have a closer-view of the bed, as with Honda's
        # setup, we need `top_mid` to be closer to the bed.
        # ------------------------------------------------------------------------------------------
  

    def new_pose_workaround(self, offsets, label, rot):
        """ Reference frame must be the `fake_ar` frame.
        
        And since that frame is fixed then everything should follow from
        earlier offsets computed by Michael.
        """
        if rot is None:
            rot = np.array([0.0, 0.0, 0.0])
        pose = np.copy(offsets)
        quat = tf.transformations.quaternion_from_euler(
                ai=rot[0], aj=rot[1], ak=rot[2])
        thread.start_new_thread(self.loop_broadcast, (pose, quat, label, 'fake_ar'))


    def loop_broadcast(self, pos, rot, name, ref):
        while True:
            self.br.sendTransform(pos, rot, rospy.Time.now(), name, ref)
            

    def make_fake_ar(self):
        """A hand-tuned y-offset value. If we follow blue tape, AR should be OK.

        If the 'fake_ar' pose is defined wrt the 'map' pose, then it should
        remain fixed throughout the robot's motion.
        """
        name = 'fake_ar'
        ref = 'map'
        thread.start_new_thread(self.loop_broadcast, (self.FAKE_AR_POS, self.FAKE_AR_ROT, name, ref))


if __name__ == "__main__":
    robot = hsrb_interface.Robot()
    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')
    tt = TableTop()

    # --------------------------------------------------------------------------
    # These two lines:
    #
    # com = COM()
    # com.go_to_initial_state(whole_body)
    #
    # Are the same as these three lines: 
    #
    # whole_body.move_to_go()
    # whole_body.move_to_joint_positions({'head_pan_joint': 1.5})
    # whole_body.move_to_joint_positions({'head_tilt_joint': -0.8})
    #
    # where Michael set the pan (rotate neck) to be about 90 degrees (1.57 rad,
    # but he used 1.5 in his code) and the tilt to be about 45 degrees (0.785 or
    # maybe the negative of that, but he used -0.8 in his code).
    # 
    # But to match Fetch, we need HSR a little taller. So, it's a four step process.
    # First, set arm to go outwards to avoid collisions. Then, rotate head (so hand is 
    # not blocking camera view), and then tilt of 45 degrees to match Fetch. Finally,
    # lift the HSR by using the `arm_lift_joint`. Verify with `rosrun tf tf_echo`.
    # --------------------------------------------------------------------------
    whole_body.move_to_go()
    whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
    whole_body.move_to_joint_positions({'head_pan_joint': np.pi/2.0})
    whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})
    whole_body.move_to_joint_positions({'arm_lift_joint': 0.120})
    # --------------------------------------------------------------------------

    # Make a fake AR marker and make table with my ugly workaround ...
    # Technically we don't even need that AR as everything is wrt map, but this
    # makes it easier to match with Michael's earlier code.
    tt.make_fake_ar()
    tt.find_table_workaround(robot)

    # Can inspect with `rosecho` commands if HSR head is at correct height & angle.
    print("Done, rospy.spin() now ...")
    rospy.spin()

    # Note: when testing movement, probably a good idea to reset the joints.
    # This will work. For now right_down results in robot facing towards bottom end.
    whole_body.move_to_go()
    ## tt.move_to_pose(omni_base,'lower_mid')
    tt.move_to_pose(omni_base,'right_down')
    ## tt.move_to_pose(omni_base,'right_mid')


