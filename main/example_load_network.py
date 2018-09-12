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

# Import the grasping network and the success network.
from fast_grasp_detect.detectors.grasp_detector import GDetector
from fast_grasp_detect.detectors.tran_detector import SDetector

# In the actual deployment code, loading success is a bit more 'roundabout':
#
# from il_ros_hsr.p_pi.bed_making.net_success import Success_Net
# Success_Net(whole_body, tt, cam, omni_base, SUCC_CONFIG, BED_CFG)
#
# then `Success_Net` calls `SDetector(SUCC_CONFIG, BEC_CFG)`.
# So it's like a wrapper, and contains separate bottom/top success checks.

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
        s_cfg = BED_CFG.SUCC_CONFIG

        # YOLO_CONV will use this, so make sure we had the same settings ...
        assert g_cfg.IMAGE_SIZE == s_cfg.IMAGE_SIZE
        assert g_cfg.T_IMAGE_SIZE_W == s_cfg.T_IMAGE_SIZE_W
        assert g_cfg.T_IMAGE_SIZE_H == s_cfg.T_IMAGE_SIZE_H
        assert g_cfg.SMALLER_NET == s_cfg.SMALLER_NET
        assert g_cfg.FIX_PRETRAINED_LAYERS == s_cfg.FIX_PRETRAINED_LAYERS
        assert g_cfg.PRE_TRAINED_DIR == s_cfg.PRE_TRAINED_DIR
        assert g_cfg.ALPHA == s_cfg.ALPHA
        assert g_cfg.FILTER_SIZE == s_cfg.FILTER_SIZE
        assert g_cfg.NUM_FILTERS == s_cfg.NUM_FILTERS

        # Build YOLO net using one of the configs, load pre-trained weights.
        self.yc = YOLO_CONV(options=g_cfg)
        self.yc.load_network()

        # Build the two policies for deployment; `get_variables()` to debug.
        self.g_detector = GDetector(g_cfg, BED_CFG, yc=self.yc)
        self.s_detector = SDetector(s_cfg, BED_CFG, yc=self.yc)


    def test_grasp(self, all_cv_files):
        """Test on all the (grasping-related) data in our CV files.

        These have ALREADY had the depth image pre-processed.
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


    def test_success(self, all_cv_files):
        """Test on all the (success/transition-related) data in our CV files.

        These have ALREADY had the depth image pre-processed.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        correct = 0
        total = 0
        incorrect0 = 0  # CNN thought a ground-truth success img was actually a failure
        incorrect1 = 0  # CNN thought a ground-truth failure img was actually a success
        thinks_success = True

        for test_list in all_cv_files:
            with open(test_list, 'r') as f:
                data = pickle.load(f)
            print("loaded test data: {} (length {})".format(test_list, len(data)))

            for idx,item in enumerate(data):
                if BED_CFG.SUCC_CONFIG.USE_DEPTH:
                    d_img = np.copy(item['d_img'])
                    result = self.s_detector.predict(d_img)
                    img = d_img
                else:
                    c_img = np.copy(item['c_img'])
                    result = self.s_detector.predict(c_img)
                    img = c_img
                result = np.squeeze( np.array(result) ) # batch size is 1
                print("  prediction: {}".format(result))

                # [x1,x2] are logits, so if x1 is higher, it thinks more likely a success
                if result[0] < result[1]:
                    prediction = 1
                    thinks_success = False
                else:
                    prediction = 0
                    thinks_success = True
                targ = item['class']
                if prediction == targ:
                    correct += 1
                else:
                    if targ == 0:
                        incorrect0 += 1
                    else:
                        incorrect1 += 1
                total += 1

                cv2.putText(img,'thinks success: {}'.format(thinks_success),
                            (30,30), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow('success_net', img)
                cv2.waitKey(30)
                #time.sleep(0.5) # pause the 'video' of images :-)

        print("Correct: {} / {}  ({:.2f})".format(correct, total, float(correct)/total))
        print("Predicted failure but image was really success: {}".format(incorrect0))
        print("Predicted success but image was really failure: {}".format(incorrect1))
        print("Remember, if we predict on training data, we should do very well. :-)")


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

    # Get paths setup.
    PATH_GRASP   = '/nfs/diskstation/seita/bed-make/cache_combo_v03/'
    PATH_SUCCESS = '/nfs/diskstation/seita/bed-make/cache_combo_v03_success/'
    all_cv_files_grasp   = sorted(
            [join(PATH_GRASP,x) for x in os.listdir(PATH_GRASP) if 'cv_' in x]
    )
    all_cv_files_success = sorted(
            [join(PATH_SUCCESS,x) for x in os.listdir(PATH_SUCCESS) if 'cv_' in x]
    )

    # Test grasping, then success net.
    TestBed.test_grasp(all_cv_files_grasp)
    TestBed.test_success(all_cv_files_success)
    print("\nDone!")
