"""Example of how to load in and use both trained networks.
"""
import sys, os, IPython, cv2, time, thread, pickle
import numpy as np
import tensorflow as tf
np.set_printoptions(suppress=True, precision=3)
from os.path import join

# We need a deployment-related config. This is where we put details
# on the paths to the grasp and success network trained weights!
import il_ros_hsr.p_pi.bed_making.config_bed as BED_CFG
from fast_grasp_detect.detectors.grasp_detector import GDetector

# Finally, load YOLO stem. Assumes we load pre-trained weights. It's the same
# for both networks, so load before building to avoid duplicate TF names.
from fast_grasp_detect.core.yolo_conv_features_cs import YOLO_CONV


class Test():

    def __init__(self):
        """For testing how to deploy the policy.

        As mentioned earlier, load in YOLO here and supply it to both of the
        detectors. (For now, assume both use YOLO pre-trained, fixed stem.)
        However, this requires an fg_cfg, which could be either the grasp or the
        success one ... i.e., the cfg we used for training. So, if both used the
        YOLO net, they better use the same cfg! If only one did, then we'll use
        that one ... a lot of manual work, unfortunately.

        BTW, this will create three TensorFlow sessions, assuming we're sharing
        the YOLO stem (which is one TF session) among the two policies.
        """
        g_cfg = BED_CFG.GRASP_CONFIG
        self.yc = YOLO_CONV(options=g_cfg)
        self.yc.load_network()
        self.g_detector = GDetector(g_cfg, BED_CFG, yc=self.yc)


    def test_grasp(self, all_cv_files):
        """Test on all the (grasping-related) data in our CV files.

        IMPORTANT: These have ALREADY had the depth image pre-processed
        before the call to `g_detector.predict(d_img)`.

        If you have raw depth images you want to try, you need to do:

            (1) convert NaN to 0 using cv2.patchNaNs(img, 0.0)
            (2) run the preprocessing in the depth preprocessing script

        For (2) use:

            d_img = depth_to_net_dim(d_img, robot='Fetch')

        fast_grasp_detect.data_aug.depth_preprocess

        Note that as of August 22, the API has the robot exposed, but keeps
        the cutoff hidden, to reduce risks of different cutoffs.
        """
        L2_results = []

        for test_list in all_cv_files:
            with open(test_list, 'r') as f:
                data = pickle.load(f)
            print("loaded test data: {} (length {})".format(test_list, len(data)))

            for idx,item in enumerate(data):
                if BED_CFG.GRASP_CONFIG.USE_DEPTH:
                    d_img = np.copy(item['d_img'])
                    result = self.g_detector.predict(d_img, draw_cross_hair=True)
                else:
                    c_img = np.copy(item['c_img'])
                    result = self.g_detector.predict(c_img, draw_cross_hair=True)
                result = np.squeeze( np.array(result) ) # batch size is 1
                targ = item['pose']
                L2 = np.sqrt( (result[0]-targ[0])**2 + (result[1]-targ[1])**2 )
                print("  prediction {} for {}, pixel L2 {:.1f}".format(result, idx, L2))
                L2_results.append(L2)
                #time.sleep(0.5) # pause the 'video' of images :-)

        print("L2s: {:.1f} +/- {:.1f}".format(np.mean(L2_results), np.std(L2_results)))


def get_variables():
    print("")
    variables = tf.trainable_variables()
    numv = 0
    for vv in variables:
        numv += np.prod(vv.shape)
        print(vv)
    print("\nNumber of parameters: {}".format(numv))


if __name__ == "__main__":
    TestBed = Test()
    PATH_GRASP   = '/nfs/diskstation/seita/bed-make/cache_combo_v01/'
    all_cv_files_grasp   = sorted(
            [join(PATH_GRASP,x) for x in os.listdir(PATH_GRASP) if 'cv_' in x]
    )
    TestBed.test_grasp(all_cv_files_grasp)
    print("\nDone!")
