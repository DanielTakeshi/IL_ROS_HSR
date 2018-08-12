import sys, cv2, time, IPython, tf, rospy, thread, hsrb_interface, geometry_msgs
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
from image_geometry import PinholeCameraModel as PCM
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, TransformStamped
from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
from tf import TransformListener, TransformBroadcaster
import numpy as np
import numpy.linalg as LA

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
OFFSET_T = 0.34

# Used for multiple offsets regarding the _trajectory_ the HSR takes.
OFFSET = 0.5


class TableTop():

    def __init__(self):
        self.tl = TransformListener()
        self.br = TransformBroadcaster()
    

    def broadcast_pose(self,pose,label):
        while True:
             self.br.sendTransform(pose['trans'], pose['quat'], rospy.Time.now(), label, 'map')
    

    def move_to_pose(self,base,label):
        base.move(geometry.pose(), 500.0, ref_frame_id=label)


    def cal_transform(self,offsets,rot = None):
        L_t_trans = tf.transformations.translation_matrix(offsets)
        M_t_L = np.matmul(self.M_t_A,L_t_trans)

        if not (rot is None):
            q_rot = tf.transformations.quaternion_from_euler(ai=rot[0],aj=rot[1],ak=rot[2])
            L_t_rot = tf.transformations.quaternion_matrix(q_rot)
            L_t_rot[:,3] = L_t_trans[:,3]
            M_t_L = np.matmul(self.M_t_A,L_t_rot)

        trans = tf.transformations.translation_from_matrix(M_t_L)
        quat = tf.transformations.quaternion_from_matrix(M_t_L)
        return trans, quat


    def make_new_pose(self,offsets,label,rot = None):
        t,q = self.cal_transform(offsets,rot = rot)
        # top_corner_trans[1] = top_corner_trans[1] + (2*OFFSET+TABLE_WIDTH)
        # top_corner_trans[0] = top_corner_trans[0] + (OFFSET+TABLE_LENGTH/2.0)
        pose = {}
        pose['trans'] = t
        pose['quat'] = q
        thread.start_new_thread(self.broadcast_pose,(pose,label))


    def calculat_ar_in_map(self,obj):
        """Gets AR and map frames set up.
        
        Make the frames wrt the AR marker, but need a transformation involving
        `map` because I think the latter will stay fixed throughout but the AR
        marker will move as the two stereo cameras move.
        """
        ar_pose = obj.get_pose(ref_frame_id = 'ar_marker/11')
        marker_pose = obj.get_pose(ref_frame_id = 'map')
        
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


    def find_table(self,robot):
        """Creates the various poses that we need for bed-making.

        Unfortunately, the robot detector seems to be malfunctioning. Here's the docs:
        https://docs.hsr.io/archives/manual/1710/en/reference/python.html#objectdetector

        Angles are in radians, so 1.57 is approx pi/2 or 90 degrees.

        Must be wrt AR marker, so e.g., a negative x-axis means moving closer to
        the HEAD up/down sides of the bed, a negative y-axis offset means moving
        AWAY from the bed, and a negative z-axis means moving UPWARDS.
        """
        detector = robot.get("marker")
        sd = detector.get_objects()
        print("detector: {}".format(detector))
        print("detector.get_objects(): {}".format(sd))
        trans, quat = self.calculat_ar_in_map(sd[0])
        
        # ---- The four corners of the bed, or more accurately, grasp targets. ----

        # HEAD DOWN, corner for first (bottom) side of bed, but has offset for grasp target.
        offsets = np.array([-(TABLE_LENGTH/2.0+0.02), OFFSET_T+0.04, -TABLE_HEIGHT+0.04])
        rot = np.array([0.0,0.0,1.57])
        self.make_new_pose(offsets,'head_down',rot = rot)

        # HEAD UP, corner for second (top) side of the bed, but has offset for grasp target.
        # As expected, only difference with HEAD UP is that we add TABLE_WIDTH to the y-axis.
        offsets = np.array([-(TABLE_LENGTH/2.0+0.02), (OFFSET_T+TABLE_WIDTH+0.02), -TABLE_HEIGHT+0.04])
        rot = np.array([0.0,0.0,-1.57])
        self.make_new_pose(offsets,'head_up',rot = rot)

        # BOTTOM DOWN AT TABLE HEIGHT.
        offsets = np.array([(TABLE_LENGTH/2.0+0.08), OFFSET_T+0.04, -TABLE_HEIGHT+TABLE_OFFSET])
        rot = np.array([0.0,0.0,1.57])
        self.make_new_pose(offsets,'bottom_down',rot = rot)

        # BOTTOM UP AT TABLE HEIGHT
        offsets = np.array([(TABLE_LENGTH/2.0+0.08), (OFFSET_T+TABLE_WIDTH+0.02), -TABLE_HEIGHT+TABLE_OFFSET])
        rot = np.array([0.0,0.0,-1.57])
        self.make_new_pose(offsets,'bottom_up',rot = rot)

        # ---- The trajectory of the HSR. ----

        # LOWER MID, where HSR begins, ideally we can see AR marker 11 from here.
        offsets = np.array([0.0,-OFFSET-0.07,0.0])
        rot = np.array([0.0,0.0,-3.14])
        self.make_new_pose(offsets,'lower_mid',rot=rot)

        # LOWER MID, HSR starts bed-making by moving here from `lower_mid`, so
        # it moves closer to the bed, and then receives the image of the setup.
        offsets = np.array([0.0,-OFFSET+0.16,0.0])
        rot = np.array([0.0,0.0,1.57])
        self.make_new_pose(offsets,'lower_start',rot=rot)

        # RIGHT CORNER, go from `lower_start` to `right_down` after we finish grasp.
        offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0), 0.0, 0.0])
        self.make_new_pose(offsets,'right_down')

        # RIGHT MID, go from `right_down` to `right_mid`. (This might get skipped, actually)
        offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0),(OFFSET+TABLE_WIDTH/2.0),0.0])
        self.make_new_pose(offsets,'right_mid')
 
        # TOP CORNER, go from `right_mid` to `right_up`.
        offsets = np.array([-(OFFSET+TABLE_LENGTH/2.0), (2*OFFSET+TABLE_WIDTH), 0.0])
        self.make_new_pose(offsets,'right_up')

        # TOP MID, where HSR goes to see the bed from top side, and performs grasp.
        offsets = np.array([0.0, (2*OFFSET+TABLE_WIDTH), 0.0])
        rot = np.array([0.0,0.0,-1.57])
        self.make_new_pose(offsets,'top_mid',rot=rot)

        # Then after this, we go in reverse, to `right_up`, etc., all the way
        # back to `lower_mid`, NOT `lower_start`!!!
        #
        # If we want the robot to have a closer-view of the bed, as with Honda's
        # setup, we need `top_mid` to be closer to the bed.
  
        # ---- Not sure when we use these? ----

        # TOP MID FAR
        offsets = np.array([0.0, 3 * OFFSET + TABLE_WIDTH, 0.0])
        rot = np.array([0.0,0.0,1.57])
        self.make_new_pose(offsets,'top_mid_far',rot = rot)

        # TOP LEFT FAR 
        offsets = np.array([(TABLE_WIDTH/2.0), 3 * OFFSET + TABLE_WIDTH, 0.0])
        rot = np.array([0.0, 0.0, 3.14 * 1.0/4.0])
        self.make_new_pose(offsets, 'top_left_far',rot = rot)


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
    # But to match Fetch, we need HSR a little taller. So let's set the height
    # first, and then we can get pan and tilt.
    # --------------------------------------------------------------------------
    whole_body.move_to_go()
    #whole_body.move_to_joint_positions({'head_tilt_joint': ...}) # NEW
    whole_body.move_to_joint_positions({'head_pan_joint': 1.5})
    whole_body.move_to_joint_positions({'head_tilt_joint': -0.8})
    # --------------------------------------------------------------------------

    tt.find_table(robot)

    # Can inspect with `rosecho` commands if HSR head is at correct height & angle.
    print("Done, rospy.spin() now ...")
    time.sleep(5)
    rospy.spin()

    ## tt.move_to_pose(omni_base,'lower_mid')
    ## tt.move_to_pose(omni_base,'right_corner')
    ## tt.move_to_pose(omni_base,'right_mid')
