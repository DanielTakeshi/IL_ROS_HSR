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
from il_ros_hsr.core.python_labeler import Python_Labeler
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.p_pi.bed_making.check_success import Success_Check
from il_ros_hsr.p_pi.bed_making.get_success import get_success
from il_ros_hsr.p_pi.bed_making.initial_state_sampler import InitialSampler
import cPickle as pickle
import numpy as np
import numpy.linalg as LA
from numpy.random import normal
from os.path import join
FAST_PATH = 'fast_path/'


def makedirs():
    if not os.path.exists(FAST_PATH):
        os.makedirs(FAST_PATH)
    pth = join(FAST_PATH,'b_grasp')
    if not os.path.exists(pth):
        os.makedirs(pth)
    pth = join(FAST_PATH,'t_grasp')
    if not os.path.exists(pth):
        os.makedirs(pth)
    pth = join(FAST_PATH,'b_success')
    if not os.path.exists(pth):
        os.makedirs(pth)
    pth = join(FAST_PATH,'t_success')
    if not os.path.exists(pth):
        os.makedirs(pth)


def red_contour(image):
    """The HSR (and Fetch) have images in BGR mode.

    Courtesy of Ron Berenstein. We tried HSV-based methods but it's really bad.
    In my cv2 version of 3.3.1, there are 3 return arguments from `findContours`.
    """
    b, g, r = cv2.split(image)
    bw0 = (r[:,:]>150).astype(np.uint8)*255

    bw1 = cv2.divide(r, g[:, :] + 1)
    bw1 = (bw1[:, :] > 1.5).astype(np.uint8)*255
    bw1 = np.multiply(bw1, bw0).astype(np.uint8) * 255
    bw2 = cv2.divide(r, b[:,:]+1)
    bw2 = (bw2[:, :] > 1.5).astype(np.uint8)*255

    bw = np.multiply(bw1, bw2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.dilate(bw, kernel, iterations=1)
    _, bw = cv2.threshold(bw,0,255,0)

    # Now get the actual contours.  Note that contour detection requires a
    # single channel image. Also, we only want the max one as that should be
    # where the sewn patch is located.
    (_, cnts, _) = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_largest = max(cnts, key = lambda cnt: cv2.contourArea(cnt))

    # Find the centroid in _pixel_space_. Draw it.
    try:
        M = cv2.moments(cnt_largest)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        peri = cv2.arcLength(cnt_largest, True)
        approx = cv2.approxPolyDP(cnt_largest, 0.02*peri, True)
        #cv2.circle(image, (cX,cY), 50, (0,0,255))
        #cv2.drawContours(image, [approx], -1, (0,255,0), 2)
        #cv2.putText(img=image, 
        #            text="{},{}".format(cX,cY), 
        #            org=(cX+10,cY+10), 
        #            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #            fontScale=1, 
        #            color=(255,255,255), 
        #            thickness=2)
        return (cX,cY)
    except:
        print("PROBLEM CANNOT DETECT CONTOUR ...")



class BedMaker():

    def __init__(self):
        """
        For faster data collection where we manually simulate it.
        We move with our hands.  This will give us the large datasets we need.

        Supports both grasping and success net data collection. If doing the
        grasping, DON'T MAKE IT A SUCCESS CASE where the blanket is all the way
        over the corner. That way we can use the images for both grasping and
        as failure cases for the success net.
        
        For the success net data collection, collect data at roughly a 5:1 ratio
        of successes:failures, and make failures the borderline cases. Then we
        borrow data from the grasping network to make it 5:5 or 1:1 for the actual
        success net training process (use another script for forming the data).
        We use the keys on the joystick to indicate the success/failure class.
        """
        makedirs()
        self.robot = robot = hsrb_interface.Robot()
        self.rgbd_map = RGBD2Map()
        self.omni_base = self.robot.get('omni_base')
        self.whole_body = self.robot.get('whole_body')
        self.cam = RGBD()
        self.com = COM()
        self.wl = Python_Labeler(cam=self.cam)

        # ----------------------------------------------------------------------
        # PARAMETERS TO CHANGE  (well, really the 'side' and 'grasp' only).
        # We choose a fixed side and collect data from there, no switching.
        # Automatically saves based on `r_count` and counting the saved files.
        # `self.grasp` remains FIXED in the code, so we're either only
        # collecting grasp or only collecting success images.
        # ----------------------------------------------------------------------
        self.side = 'BOTTOM'    # CHANGE AS NEEDED
        self.grasp = False      # CHANGE AS NEEDED
        self.grasp_count = 0
        self.success_count = 0
        self.true_count = 0
        self.r_count = self.get_rollout_number()
        self.joystick = JoyStick_X(self.com)
        print("NOTE: grasp={} (success={}), side: {}, rollout num: {}".format(
                self.grasp, not self.grasp, self.side, self.r_count))
        print("Press X for any SUCCESS (class 0), Y for FAILURES (class 1).")

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

        For now, let's save based on how many `data.pkl` files we have in the
        appropriate directory.
        """
        if self.side == "BOTTOM":
            nextdir = 'b_grasp'
            if not self.grasp:
                nextdir = 'b_success'
            rollouts = sorted(
                [x for x in os.listdir(join(FAST_PATH,nextdir)) if 'data' in x and 'pkl' in x]
            )
        else:
            nextdir = 't_grasp'
            if not self.grasp:
                nextdir = 't_success'
            rollouts = sorted(
                [x for x in os.listdir(join(FAST_PATH,nextdir)) if 'data' in x and 'pkl' in x]
            )
        return len(rollouts)


    def position_head(self):
        """Ah, I see, we can go straight to the top. Whew.

        It's new code reflecting the different poses and HSR joints:
        But, see important note below about commenting out four lines ...
        """
        self.whole_body.move_to_go()
        if self.side == "BOTTOM":
            self.tt.move_to_pose(self.omni_base,'lower_start_tmp')
        else:
            self.tt.move_to_pose(self.omni_base,'right_down')
            self.tt.move_to_pose(self.omni_base,'right_up')
            self.tt.move_to_pose(self.omni_base,'top_mid_tmp')
        # NOTE: If robot is already at the top, I don't want to adjust position.
        # Thus, comment out the three lines and the preceding `else` statement.

        self.whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
        self.whole_body.move_to_joint_positions({'head_pan_joint':  np.pi/2.0})
        self.whole_body.move_to_joint_positions({'arm_lift_joint':  0.120})
        self.whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0}) # -np.pi/36.0})


    def collect_data_grasp_only(self):
        """Collect data for the grasping network only, like H's method.

        Actually, some of these images should likely be part of the success
        network training, where the 'success=False' because I don't think I
        collected data here that was considered a 'success=True' ...
        """
        data = [] 
        assert self.grasp
        rc = str(self.r_count).zfill(3)

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
                self.grasp_count += 1

                # Add to dictionary info we want, including target pose.
                # Also add 'type' key since data augmentation code uses it.
                pose = red_contour(c_img)
                info = {'c_img':c_img, 'd_img':d_img, 'pose':pose, 'type':grasp}
                data.append(info)
                print("  image {}, pose: {}".format(len(data), pose))

                # --------------------------------------------------------------
                # We better save each time since we might get a failure to
                # detect, thus losing some data. We overwrite existing saved
                # files, which is fine since it's the current rollout `r_count`.
                # Since we detect pose before this, if the pose isn't detected,
                # we don't save. Good.
                # --------------------------------------------------------------
                if self.side == 'BOTTOM':
                    save_path = join(FAST_PATH, 'b_grasp', 'data_{}.pkl'.format(rc))
                else:
                    save_path = join(FAST_PATH, 't_grasp', 'data_{}.pkl'.format(rc))
                with open(save_path, 'w') as f:
                    pickle.dump(data, f)

            # Kill the script and re-position HSR to get diversity in camera views.
            if (cur_recording[1] < -0.1 and self.true_count%20 == 0):
                print("ROLLOUT DONE (cur_recording: {})".format(cur_recording))
                print("Length is {}. See our saved pickle files.".format(len(data)))
                sys.exit()

            # Necessary, otherwise we'd save 3-4 times per click.
            self.true_count += 1


    def collect_data_success_only(self):
        """Collect data for the success network.
        
        Should be more emphasis on the success cases (not failures) because the
        grasing network data can supplement the failures. Focus on _borderline_
        failures in this method.

        Recall that 0 = successful grasp, 1 = failed grasp.

        SAVE AND EXIT FREQUENTLY, perhaps after every 15-20 images. It's easy to
        make a mistake with the class label, so better to exit early often.
        """
        data = [] 
        assert not self.grasp
        rc = str(self.r_count).zfill(3)

        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            cv2.imshow('video_feed', c_img)
            cv2.waitKey(30)
            cur_recording = self.joystick.get_record_actions_passive()

            # Joystick controllers. Y is top, and going counterclockwise:
            # Y: [ 1,  0] # FAILURE images (class index 1)
            # X: [-1,  0] # SUCCESS images (class index 0)
            # A: [ 0,  1]
            # B: [ 0, -1] # terminate data collection
            # ------------------------------------------------------------------
            if (cur_recording[0] < -0.1 or cur_recording[0] > 0.1) and self.true_count % 20 == 0:
                print("PHOTO SNAPPED (cur_recording: {})".format(cur_recording))
                if cur_recording[0] < -0.1:
                    s_class = 0
                elif cur_recording[0] > 0.1:
                    s_class = 1
                else:
                    raise ValueError(cur_recording)
                self.save_image(c_img, d_img, success_class=s_class)
                self.success_count += 1

                # Add to dictionary info we want, including the class.
                info = {'c_img':c_img, 'd_img':d_img, 'class':s_class, 'type':'success'}
                data.append(info)
                print("  image {}, class: {}".format(len(data), s_class))

                if self.side == 'BOTTOM':
                    save_path = join(FAST_PATH, 'b_success', 'data_{}.pkl'.format(rc))
                else:
                    save_path = join(FAST_PATH, 't_success', 'data_{}.pkl'.format(rc))
                with open(save_path, 'w') as f:
                    pickle.dump(data, f)

            # Kill the script and re-position HSR to get diversity in camera views.
            if (cur_recording[1] < -0.1 and self.true_count % 20 == 0):
                print("ROLLOUT DONE (cur_recording: {})".format(cur_recording))
                print("Length is {}. See our saved pickle files.".format(len(data)))
                sys.exit()

            # Necessary, otherwise we'd save 3-4 times per click.
            self.true_count += 1


    def save_image(self, c_img, d_img, success_class=None):
        """Save images. Don't forget to process depth images.

        For now I'm using a tuned cutoff like 1400, at least to _visualize_.
        NOTE: since the cutoff for turning depth images into black may change,
        it would be better to save the original d_img in a dictionary. Don't use
        cv2.imwrite() as I know from experience that it won't work as desired.
        """
        rc = str(self.r_count).zfill(3)
        f_rc_grasp   = 'frame_{}_{}.png'.format(rc, str(self.grasp_count).zfill(2))
        f_rc_success = 'frame_{}_{}_class_{}.png'.format(rc,
                str(self.success_count).zfill(2), success_class)
        if np.isnan(np.sum(d_img)):
            cv2.patchNaNs(d_img, 0.0)
        d_img = depth_to_net_dim(d_img, cutoff=1400) # for visualization only

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
    #cp.collect_data_grasp_only()
    cp.collect_data_success_only()
