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

# Don't forget analytic versions, but we'll use the one for grasping.
from il_ros_hsr.p_pi.bed_making.analytic_grasp import Analytic_Grasp
# AH name clash!!
#from il_ros_hsr.p_pi.bed_making.analytic_success import Success_Net
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

        We use all three variants (analytic, human, networks) here due to
        similarities in code structure.
        """
        self.args = args
        DEBUG = True

        # Set up the robot.
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
        self.b_grasp_count = 0
        self.t_grasp_count = 0

        # AH, build the YOLO network beforehand.
        g_cfg = BED_CFG.GRASP_CONFIG
        s_cfg = BED_CFG.SUCC_CONFIG
        self.yc = YOLO_CONV(options=s_cfg)
        self.yc.load_network()

        # Policy for grasp detection, using Deep Imitation Learning.
        # Or, actually, sometimes we will use humans or an analytic version.
        if DEBUG:
            self._test_variables()
        print("\nnow forming the GDetector with type {}".format(args.g_type))
        if args.g_type == 'network':
            self.g_detector = GDetector(g_cfg, BED_CFG, yc=self.yc)
        elif args.g_type == 'analytic':
            self.g_detector = Analytic_Grasp() # TODO not implemented!
        elif args.g_type == 'human':
            print("Using a human, don't need to have a `g_detector`. :-)")

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

        # For evaluating coverage.
        self.img_start = None
        self.img_final = None
        self.img_start2 = None
        self.img_final2 = None


        # For grasp offsets.
        self.apply_offset = False


    def bed_make(self):
        """Runs the pipeline for deployment, testing out bed-making.
        """
        # Get the starting image (from USB webcam). Try a second as well.
        cap = cv2.VideoCapture(0)
        frame = None
        while frame is None:
            ret, frame = cap.read()
            cv2.waitKey(50)
        self.image_start = frame
        cv2.imwrite('image_start.png', self.image_start)

        _, frame = cap.read()
        self.image_start2 = frame
        cv2.imwrite('image_start2.png', self.image_start2)

        cap.release()
        print("NOTE! Recorded `image_start` for coverage evaluation. Was it set up?")

        def get_pose(data_all):
            # See `find_pick_region_labeler` in `p_pi/bed_making/gripper.py`.
            # It's because from the web labeler, we get a bunch of objects.
            # So we have to compute the pose (x,y) from it.
            res = data_all['objects'][0]
            x_min = float(res['box'][0])
            y_min = float(res['box'][1])
            x_max = float(res['box'][2])
            y_max = float(res['box'][3])
            x = (x_max - x_min)/2.0 + x_min
            y = (y_max - y_min)/2.0 + y_min
            return (x,y)

        args = self.args
        use_d = BED_CFG.GRASP_CONFIG.USE_DEPTH
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

                # --------------------------------------------------------------
                # Process depth images! Helps network, human, and (presumably) analytic.
                # Obviously human can see the c_img as well ... hard to compare fairly.
                # --------------------------------------------------------------
                if use_d:
                    if np.isnan(np.sum(d_img)):
                        cv2.patchNaNs(d_img, 0.0)
                    d_img = depth_to_net_dim(d_img, robot='HSR')
                    policy_input = np.copy(d_img)
                else:
                    policy_input = np.copy(c_img)

                # --------------------------------------------------------------
                # Run grasp detector to get data=(x,y) point for target, record stats.
                # Note that the web labeler returns a dictionary like this:
                # {'objects': [{'box': (155, 187, 165, 194), 'class': 0}], 'num_labels': 1}
                # but we really want just the 2D grasping point. So use `get_pose()`.
                # Also, for the analytic one, we'll pick the highest point ourselves.
                # --------------------------------------------------------------
                sgraspt = time.time()
                if args.g_type == 'network':
                    data = self.g_detector.predict(policy_input)
                elif args.g_type == 'analytic':
                    data_all = self.wl.label_image(policy_input)
                    data = get_pose(data_all)
                elif args.g_type == 'human':
                    data_all = self.wl.label_image(policy_input)
                    data = get_pose(data_all)
                egraspt = time.time()

                g_predict_t = egraspt - sgraspt
                print("Grasp predict time: {:.2f}".format(g_predict_t))
                self.record_stats(c_img, d_img_raw, data, self.side, g_predict_t, 'grasp')

                # For safety, we can check image and abort as needed before execution.
                if use_d:
                    img = self.dp.draw_prediction(d_img, data)
                else:
                    img = self.dp.draw_prediction(c_img, data)
                caption = 'G Predicted: {} (ESC to abort, other key to proceed)'.format(data)
                call_wait_key( cv2.imshow(caption,img) )

                # --------------------------------------------------------------
                # Broadcast grasp pose, execute the grasp, check for success.
                # We'll use the `find_pick_region_net` since the `data` is the
                # (x,y) pose, and not `find_pick_region_labeler`.
                # --------------------------------------------------------------
                self.gripper.find_pick_region_net(pose=data,
                                                  c_img=c_img,
                                                  d_img=d_img_raw,
                                                  count=self.grasp_count,
                                                  side=self.side,
                                                  apply_offset=self.apply_offset)
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
                self.check_success_state(policy_input)


    def check_success_state(self, old_grasp_image):
        """
        Checks whether a single grasp in a bed-making trajectory succeeded.
        Depends on which side of the bed the HSR is at. Invokes the learned
        success network policy and transitions the HSR if successful.

        When we record the data, c_img and d_img should be what success net saw.

        UPDATE: now we can pass in the previous `d_img` from the grasping to
        compare the difference. Well, technically the `policy_input` so it can
        handle either case.
        """
        use_d = BED_CFG.SUCC_CONFIG.USE_DEPTH
        if self.side == "BOTTOM":
            result = self.sn.check_bottom_success(use_d, old_grasp_image)
            self.b_grasp_count += 1
        else:
            result = self.sn.check_top_success(use_d, old_grasp_image)
            self.t_grasp_count += 1
        self.grasp_count += 1
        assert self.grasp_count == self.b_grasp_count + self.t_grasp_count

        success     = result['success']
        data        = result['data']
        c_img       = result['c_img']
        d_img       = result['d_img']
        d_img_raw   = result['d_img_raw']
        s_predict_t = result['s_predict_t']
        img_diff    = result['diff_l2']
        img_ssim    = result['diff_ssim']
        self.record_stats(c_img, d_img_raw, data, self.side, s_predict_t, 'success')

        # We really need a better metric, such as 'structural similarity'.
        # Edit: well, it's probably marginally better, I think.
        # I use an L2 threshold of 98k, and an SSIM threshold of 0.88.

        if BED_CFG.GRASP_CONFIG.USE_DEPTH != BED_CFG.SUCC_CONFIG.USE_DEPTH:
            print("grasp vs success for using depth differ")
            print("for now we'll ignore the offset issue.")
        else:
            print("L2 and SSIM btwn grasp & next image: {:.1f} and {:.3f}".format(
                    img_diff, img_ssim))
            if img_ssim >= 0.875 or img_diff < 85000:
                print("APPLYING OFFSET! (self.apply_offset = True)")
                self.apply_offset = True
            else:
                print("no offset applied (self.apply_offset = False)")
                self.apply_offset = False

        # Have user confirm that this makes sense.
        caption = "Success net saw this and thought: {}. Press any key".format(success)
        if use_d:
            call_wait_key( cv2.imshow(caption,d_img) )
        else:
            call_wait_key( cv2.imshow(caption,c_img) )

        # Limit amount of grasp attempts per side, pretend 'success' anyway.
        lim = BED_CFG.GRASP_ATTEMPTS_PER_SIDE
        if (self.side == 'BOTTOM' and self.b_grasp_count >= lim) or \
                (self.side == 'TOP' and self.t_grasp_count >= lim):
            print("We've hit {} for this side so set success=True".format(lim))
            success = True

        # Handle transitioning to different side
        if success:
            if self.side == "BOTTOM":
                self.transition_to_top()
                self.side = 'TOP'
            else:
                self.transition_to_start()
            print("We're moving to another side so revert self.apply_offset = False.")
            self.apply_offset = False
        else:
            self.new_grasp = False


    def transition_to_top(self):
        """Transition to top (not bottom)."""
        transition_time = self.move_to_top_side()
        self.move_time_stats.append( transition_time )


    def transition_to_start(self):
        """Transition to start=bottom, SAVE ROLLOUT STATS, exit program.

        The `rollout_stats` is a list with a bunch of stats recorded via the
        class method `record_stats`. We save with a top-down webcam and save
        before moving back, since the HSR could disconnect.
        """
        # Record the final image for evaluation later (from USB webcam).
        cap = cv2.VideoCapture(0)
        frame = None
        while frame is None:
            ret, frame = cap.read()
        self.image_final = frame
        cv2.imwrite('image_final.png', self.image_final)

        _, frame = cap.read()
        self.image_final2 = frame
        cv2.imwrite('image_final2.png', self.image_final2)

        cap.release()
        print("NOTE! Recorded `image_final` for coverage evaluation.")

        # Append some last-minute stuff to `self.rollout_stats` for saving.
        final_stuff = {
            'image_start': self.image_start,
            'image_final': self.image_final,
            'image_start2': self.image_start2,
            'image_final2': self.image_final2,
            'grasp_times': self.g_time_stats,
            'move_times': self.move_time_stats,
            'args': self.args, # ADDING THIS! Now we can 'retrace' our steps.
        }
        self.rollout_stats.append(final_stuff)

        # SAVE, move to start, then exit.
        self.com.save_stat(self.rollout_stats, target_path=self.args.save_path)
        self.move_to_start()
        sys.exit()


    def record_stats(self, c_img, d_img, data, side, time, typ):
        """Adds a dictionary to the `rollout_stats` list.

        We can tell it's a 'net' thing due to 'net_pose' and 'net_succ' keys.
        EDIT: argh wish I hadn't done that since this script also handles the
        human and analytic cases. Oh well, too late for that now.
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
        tic = time.time()
        self.tt.move_to_pose(self.omni_base,'right_down_1')
        self.tt.move_to_pose(self.omni_base,'right_mid_1')
        self.tt.move_to_pose(self.omni_base,'right_up_1')
        self.tt.move_to_pose(self.omni_base,'top_mid_tmp')
        toc = time.time()
        return toc-tic


    def move_to_start(self):
        """Assumes we're at the top and we go back to the start.

        Go to lower_start_tmp to be at the same view as we started with, so that
        we take a c_img and compare coverage.
        """
        self.whole_body.move_to_go()
        tic = time.time()
        self.tt.move_to_pose(self.omni_base,'right_up_2')
        self.tt.move_to_pose(self.omni_base,'right_mid_2')
        self.tt.move_to_pose(self.omni_base,'right_down_2')
        self.tt.move_to_pose(self.omni_base,'lower_start_tmp')
        toc = time.time()
        return toc-tic


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


    def _test_grasp(self):
        """Simple tests for grasping. Don't forget to process depth images.

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


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--phase', type=int, help='1 for checking poses, 2 for deployment.')
    pp.add_argument('--g_type', type=str, help='must be in [network, human, analytic]')
    pp.add_argument('--use_rgb', action='store_true', default=False,
        help='If doing this, we need to use the rgb network.')
    args = pp.parse_args()

    assert args.phase in [1,2]

    if args.use_rgb:
        assert not BED_CFG.GRASP_CONFIG.USE_DEPTH
        assert args.g_type == 'network'
        args.save_path = BED_CFG.DEPLOY_NET_PATH+'_rgb'
    else:
        assert BED_CFG.GRASP_CONFIG.USE_DEPTH
        if args.g_type == 'network':
            args.save_path = BED_CFG.DEPLOY_NET_PATH
        elif args.g_type == 'human':
            args.save_path = BED_CFG.DEPLOY_HUM_PATH
        elif args.g_type == 'analytic':
            args.save_path = BED_CFG.DEPLOY_ANA_PATH
        else:
            raise ValueError(args.g_type)
    print("\nNote that we will be saving to: {}".format(args.save_path))
    print("(double check the blanket type!)")

    cp = BedMaker(args)
    #cp._test_grasp()
    #cp._test_success()
    cp.bed_make()
    rospy.spin()
