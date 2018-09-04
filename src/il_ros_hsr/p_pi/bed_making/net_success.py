import os, Pyro4, time, IPython, cv2, hsrb_interface, time
import cPickle as pickle
import numpy as np
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from il_ros_hsr.p_pi.bed_making.gripper import Bed_Gripper
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.sensors import  RGBD
from fast_grasp_detect.detectors.tran_detector import SDetector
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim
from skimage.measure import compare_ssim


class Success_Net:
    """The success network policy, check success by calling network.

    It's similar to `check_success.py` except for the DNN version.
    Actual net, btw, is in `fast_grasp_detect.detectors.tran_detector`.
    """

    def __init__(self, whole_body, tt, cam, base, fg_cfg, bed_cfg, yc):
        self.whole_body = whole_body
        self.tt = tt
        self.cam = cam
        self.omni_base = base
        self.sdect = SDetector(fg_cfg=fg_cfg, bed_cfg=bed_cfg, yc=yc)


    def check_bottom_success(self, use_d, old_grasp_image):
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base, 'lower_start_tmp')
        result = self.shared_code(use_d, old_grasp_image)
        return result


    def check_top_success(self, use_d, old_grasp_image):
        self.whole_body.move_to_go()
        self.tt.move_to_pose(self.omni_base, 'top_mid_tmp')
        result = self.shared_code(use_d, old_grasp_image)
        return result


    def shared_code(self, use_d, old_grasp_image):
        """Shared code for calling the success network.

        For the timing, avoid counting the time for processing images.

        Returns dictionary with a bunch of info for later.
        """
        self.whole_body.move_to_joint_positions({'arm_flex_joint': -np.pi/16.0})
        self.whole_body.move_to_joint_positions({'head_pan_joint':  np.pi/2.0})
        self.whole_body.move_to_joint_positions({'arm_lift_joint':  0.120})
        self.whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})# -np.pi/36.0})

        # Retrieve and (if necessary) process images.
        time.sleep(3)
        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()
        d_img_raw = np.copy(d_img)
        if use_d:
            d_img = depth_to_net_dim(d_img, robot='HSR')
            img = np.copy(d_img)
        else:
            img = np.copy(c_img)

        # Call network and record time.
        stranst = time.time()
        data = self.sdect.predict(img)
        etranst = time.time()
        s_predict_t = etranst - stranst
        print("\nSuccess predict time: {:.2f}".format(s_predict_t))
        print("Success net output (i.e., logits): {}\n".format(data))

        # The success net outputs a 2D result for a probability vector.
        ans = np.argmax(data)
        success = (ans == 0)

        # NEW! Can also tell us the difference between grasp and success imgs.
        diff = np.linalg.norm( old_grasp_image - img )
        score = compare_ssim( old_grasp_image[:,:,0], img[:,:,0] )

        result = {
            'success': success,
            'data': data,
            'c_img': c_img,
            'd_img': d_img,
            'd_img_raw': d_img_raw,
            's_predict_t': s_predict_t,
            'diff_l2': diff,
            'diff_ssim': score,
        }
        return result


if __name__ == "__main__":
    robot = hsrb_interface.Robot()
    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')
    cam = RGBD()
    wl = Python_Labeler(cam)
    sc = Success_Check(whole_body,None,cam,omni_base)
    print "RESULT ", sc.check_success(wl)
