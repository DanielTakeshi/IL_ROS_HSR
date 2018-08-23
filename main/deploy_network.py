import IPython, os, sys, cv2, time, thread, rospy, glob, hsrb_interface, argparse
from tf import TransformBroadcaster, TransformListener
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped, Twist
import il_ros_hsr.p_pi.bed_making.config_bed as BED_CFG
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
from fast_grasp_detect.labelers.online_labeler import QueryLabeler
import tensorflow as tf
import numpy as np
import numpy.linalg as LA
from numpy.random import normal

# Grasping and success networks (success is a bit more 'roundabout' but w/e).
# We also need the depth preprocessing code for before we feed image to detector.
# And we need YOLO network to build the common shared weights beforehand.

from fast_grasp_detect.detectors.grasp_detector import GDetector
from il_ros_hsr.p_pi.bed_making.net_success import Success_Net
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim
from fast_grasp_detect.core.yolo_conv_features_cs import YOLO_CONV
from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction
ESC_KEYS = [27, 1048603]


def call_wait_key(nothing=None):
    """Call this like: `call_wait_key( cv2.imshow(...) )`."""
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        print("Pressed ESC key. Terminating program...")
        sys.exit()


class BedMaker():

    def __init__(self, args):
        """For deploying the bed-making policy, not for data collection.

        Uses the neural network, `GDetector`, not the analytic baseline.
        """
        DEBUG = True
        self.robot = robot = hsrb_interface.Robot()
        if DEBUG:
            print("finished: hsrb_interface.Robot()...")
        self.rgbd_map = RGBD2Map()
        self.omni_base = self.robot.get('omni_base')
        if DEBUG:
            print("finished: robot.get(omni_base)...")
        self.whole_body = self.robot.get('whole_body')
        if DEBUG:
            print("finished: robot.get(whole_body)...")
        self.cam = RGBD()
        self.com = COM()
        self.wl = Python_Labeler(cam=self.cam)

        # Set up initial state, table, etc. Don't forget view mode!
        self.view_mode = BED_CFG.VIEW_MODE
        self.com.go_to_initial_state(self.whole_body)
        if DEBUG:
            print("finished: go_to_initial_state() ...")
        self.tt = TableTop()
        if DEBUG:
            print("finished: TableTop()...")

        # For now, a workaround. Ugly but it should do the job ...
        #self.tt.find_table(robot)
        self.tt.make_fake_ar()
        self.tt.find_table_workaround(robot)

        #self.ins = InitialSampler(self.cam)
        self.side = 'BOTTOM'
        self.grasp_count = 0

        # AH, build the YOLO network beforehand.
        g_cfg = BED_CFG.GRASP_CONFIG
        s_cfg = BED_CFG.SUCC_CONFIG
        self.yc = YOLO_CONV(options=g_cfg)
        self.yc.load_network()

        # Policy for grasp detection, using Deep Imitation Learning.
        if DEBUG:
            self._test_variables()
            print("\nnow forming the GDetector")
        self.g_detector = GDetector(g_cfg, BED_CFG, yc=self.yc)
        
        if DEBUG:
            self._test_variables()
            print("\nnow making success net")
        self.sn = Success_Net(self.whole_body, self.tt, self.cam,
                self.omni_base, fg_cfg=s_cfg, bed_cfg=BED_CFG, yc=self.yc)

        # Bells and whistles.
        self.br = TransformBroadcaster()
        self.tl = TransformListener()
        self.gp = GraspPlanner()
        self.gripper = Bed_Gripper(self.gp, self.cam, self.com.Options, robot.get('gripper'))
        self.dp = DrawPrediction()

        # When we start, do rospy.spin() to check the frames (phase 1). Then re-run.
        # The current hack we have to get around crummy AR marker detection. :-(
        if DEBUG:
            self._test_variables()
        print("Finished with init method")
        time.sleep(4)
        if args.phase == 1:
            print("Now doing rospy.spin() because phase = 1.")
            rospy.spin()


    def _test_grasp(self):
        """Simple tests for grasping net. Don't forget to process depth images.

        Do this independently of any rollout ...
        """
        print("\nNow in `test_grasp` to check grasping net...")
        self.position_head()
        time.sleep(3)

        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()
        if np.isnan(np.sum(d_img)):
            cv2.patchNaNs(d_img, 0.0)
        d_img = depth_to_net_dim(d_img, robot='HSR')
        pred = self.g_detector.predict( np.copy(d_img) )
        img = self.dp.draw_prediction(d_img, pred)

        print("prediction: {}".format(pred))
        caption = 'G Predicted: {} (ESC to abort, other key to proceed)'.format(pred)
        cv2.imshow(caption, img)
        key = cv2.waitKey(0)
        if key in ESC_KEYS:
            print("Pressed ESC key. Terminating program...")
            sys.exit()


    def _test_success(self):
        """Simple tests for success net. Don't forget to process depth images.

        Should be done after a grasp test since I don't re-position...  Note: we
        have access to `self.sn` but that isn't the actual net which has a
        `predict`, but it's a wrapper (explained above), but we can access the
        true network via `self.sn.sdect` and from there call `predict`.
        """
        print("\nNow in `test_success` to check success net...")
        time.sleep(3)
        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()
        if np.isnan(np.sum(d_img)):
            cv2.patchNaNs(d_img, 0.0)
        d_img = depth_to_net_dim(d_img, robot='HSR')
        result = self.sn.sdect.predict( np.copy(d_img) )
        result = np.squeeze(result)

        print("s-net pred: {} (if [0]<[1] failure, else success...)".format(result))
        caption = 'S Predicted: {} (ESC to abort, other key to proceed)'.format(result)
        cv2.imshow(caption, d_img)
        key = cv2.waitKey(0)
        if key in ESC_KEYS:
            print("Pressed ESC key. Terminating program...")
            sys.exit()


    def _test_variables(self):
        """Test to see if TF variables were loaded correctly.
        """
        vars = tf.trainable_variables()
        print("\ntf.trainable_variables:")
        for vv in vars:
            print("  {}".format(vv))
        print("done\n")


    def bed_make(self):
        """Runs the pipeline for deployment, testing out bed-making.
        """
        self.get_new_grasp = True
        self.new_grasp = True
        self.rollout_stats = [] # What we actually save for analysis later

        # Add to self.rollout_stats at the end for more timing info
        self.g_time_stats = []      # for _execution_ of a grasp
        self.move_time_stats = []   # for moving to the other side

        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            if (not c_img.all() == None and not d_img.all() == None):
                if self.new_grasp:
                    self.position_head()
                else:
                    self.new_grasp = True
                time.sleep(3)
                c_img = self.cam.read_color_data()
                d_img = self.cam.read_depth_data()
                d_img_raw = np.copy(d_img) # Needed for determining grasp pose

                if BED_CFG.GRASP_CONFIG.USE_DEPTH:
                    if np.isnan(np.sum(d_img)):
                        cv2.patchNaNs(d_img, 0.0)
                    d_img = depth_to_net_dim(d_img, robot='HSR')
                    net_input = np.copy(d_img)
                else:
                    net_input = np.copy(c_img)

                # Run grasp detector to get data=(x,y) point for grasp target.
                sgraspt = time.time()
                data = self.g_detector.predict(net_input)
                egraspt = time.time()
                g_predict_t = egraspt - sgraspt
                print("Grasp predict time: {:.2f}".format(g_predict_t))
                self.record_stats(c_img, d_img, data, self.side, g_predict_t, 'grasp')

                # For safety, we can check image and abort as needed before execution.
                if BED_CFG.GRASP_CONFIG.USE_DEPTH:
                    img = self.dp.draw_prediction(d_img, data)
                else:
                    img = self.dp.draw_prediction(c_img, data)
                caption = 'G Predicted: {} (ESC to abort, other key to proceed)'.format(data)
                call_wait_key( cv2.imshow(caption,img) )

                # Broadcast grasp pose, execute the grasp, check for success.
                self.gripper.find_pick_region_net(data, c_img, d_img_raw, self.grasp_count)
                pick_found, bed_pick = self.check_card_found()

                if self.side == "BOTTOM":
                    self.whole_body.move_to_go()
                    self.tt.move_to_pose(self.omni_base,'lower_start')
                    tic = time.time()
                    self.gripper.execute_grasp(bed_pick, self.whole_body, 'head_down')
                    toc = time.time()
                else:
                    self.whole_body.move_to_go()
                    self.tt.move_to_pose(self.omni_base,'top_mid')
                    tic = time.time()
                    self.gripper.execute_grasp(bed_pick, self.whole_body, 'head_up')
                    toc = time.time()
                self.g_time_stats.append( toc-tic )
                self.check_success_state()


    def check_success_state(self):
        """
        Checks whether a single grasp in a bed-making trajectory succeeded.
        Depends on which side of the bed the HSR is at. Invokes the learned
        success network policy and transitions the HSR if successful.

        When we record the data, c_img and d_img should be what success net saw.
        """
        use_d = BED_CFG.GRASP_CONFIG.USE_DEPTH
        if self.side == "BOTTOM":
            result = self.sn.check_bottom_success(use_d)
        else:
            result = self.sn.check_top_success(use_d)
        success     = result['success']
        data        = result['data']
        c_img       = result['c_img']
        d_img       = result['d_img']
        s_predict_t = result['s_predict_t']
        self.record_stats(c_img, d_img, data, self.side, s_predict_t, 'success')

        # Have user confirm that this makes sense.
        caption = "Success net saw this and thought: {}. Press any key".format(success)
        if use_d:
            call_wait_key( cv2.imshow(caption,d_img) )
        else:
            call_wait_key( cv2.imshow(caption,c_img) )

        # Handle transitioning to different side
        if success:
            if self.side == "BOTTOM":
                self.transition_to_top()
                self.side = 'TOP'
            else:
                self.transition_to_start()
        else:
            self.new_grasp = False
        self.grasp_count += 1

        # Limit amount of grasp attempts to cfg.GRASP_OUT (was 8 by default).
        if self.grasp_count > BED_CFG.GRASP_OUT:
            self.transition_to_start()


    def transition_to_top(self):
        """Transition to top (not bottom)."""
        tic = time.time()
        self.move_to_top_side()
        toc = time.time()
        self.move_time_stats.append( toc-tic )


    def transition_to_start(self):
        """Transition to start=bottom, SAVE ROLLOUT STATS, exit program.

        The `rollout_stats` is a list with a bunch of stats recorded via the
        class method `record_stats`.
        
        Now that I think about it, we should save the final images here, so that
        we can evaluate sheet coverage before and after from the same view.
        Unfortunately it won't be exact but probably fairer to do this? We can
        also do this in addition to the top-down view.

        Then we do a `sys.exit()` here. Best to just do one trajectory per call.
        """
        self.move_to_start()
        self.side = 'BOTTOM'
        self.position_head()
        time.sleep(3)
        c_img = self.cam.read_color_data()

        # Append some last-minute stuff to `self.rollout_stats` for saving.
        final_stuff = {
            'final_c_img': c_img,
            'grasp_times': self.g_time_stats,
            'move_times': self.move_time_stats,
        }
        self.rollout_stats.append(final_stuff)

        self.com.save_stat(self.rollout_stats, target_path=BED_CFG.DEPLOY_NET_PATH)
        sys.exit()


    def record_stats(self, c_img, d_img, data, side, time, typ):
        """Adds a dictionary to the `rollout_stats` list.
        
        We can tell it's a 'net' thing due to 'net_pose' and 'net_succ' keys.
        """
        assert side in ['BOTTOM', 'TOP']
        grasp_point = {}
        grasp_point['c_img'] = c_img
        grasp_point['d_img'] = d_img
        if typ == "grasp":
            grasp_point['net_pose'] = data
            grasp_point['g_net_time'] = time
        elif typ == "success":
            grasp_point['net_succ'] = data
            grasp_point['s_net_time'] = time
        else:
            raise ValueError(typ)
        grasp_point['side'] = side
        grasp_point['type'] = typ
        self.rollout_stats.append(grasp_point)


    def position_head(self):
        """Position head for a grasp.
        
        Use lower_start_tmp so HSR looks 'sideways'; thus, hand is not in the way.
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
        self.tt.move_to_pose(self.omni_base,'right_down_1')
        self.tt.move_to_pose(self.omni_base,'right_mid_1')
        self.tt.move_to_pose(self.omni_base,'right_up_1')
        self.tt.move_to_pose(self.omni_base,'top_mid_tmp')


    def move_to_start(self):
        """Assumes we're at the top and we go back to the start.
        
        Go to lower_start_tmp to be at the same view as we started with, so that
        we take a c_img and compare coverage.
        """
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base,'right_up_2')
        self.tt.move_to_pose(self.omni_base,'right_mid_2')
        self.tt.move_to_pose(self.omni_base,'right_down_2')
        #self.tt.move_to_pose(self.omni_base,'lower_mid')
        self.tt.move_to_pose(self.omni_base,'lower_start_tmp')


    def check_card_found(self):
        """Looks up the pose for where the HSR's hand should go to."""
        transforms = self.tl.getFrameStrings()
        cards = []
        try:
            for transform in transforms:
                #print(transform)
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
    #cp._test_grasp()
    #cp._test_success()
    cp.bed_make()
    rospy.spin()
