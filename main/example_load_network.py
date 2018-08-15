"""Example of how to load in and use a trained network.
"""
import sys, os, IPython, cv2, time, thread, pickle
import numpy as np
np.set_printoptions(suppress=True, precision=3)
from os.path import join

# We need a config any time we deploy (or use IL_ROS_HSR, for that matter).
import il_ros_hsr.p_pi.bed_making.config_bed as BED_CFG

# Import the grasping network.
from fast_grasp_detect.detectors.grasp_detector import GDetector

# Something similar happens to the success network.
# TODO for now we're testing the grasping network, and we'll get the success later
#from il_ros_hsr.p_pi.bed_making.net_success import Success_Net


class Test():

    def __init__(self):
        """For testing how to deploy the policy.
        """
        self.g_detector = GDetector(fg_cfg=BED_CFG.GRASP_CONFIG, bed_cfg=BED_CFG)


    def test(self, all_cv_files):
        """Test on all the data in all our cross validation files.
        """
        L2_results = []

        for test_list in all_cv_files:
            with open(test_list, 'r') as f:
                data = pickle.load(f)
            print("loaded test data: {} (length {})".format(test_list, len(data)))

            for idx,item in enumerate(data):
                if BED_CFG.GRASP_CONFIG.USE_DEPTH:
                    d_img = np.copy(item['d_img'])
                    result = self.g_detector.predict(d_img)
                else:
                    c_img = np.copy(item['c_img'])
                    result = self.g_detector.predict(c_img)
                result = np.array(result)
                targ = item['pose']
                L2 = np.sqrt( (result[0]-targ[0])**2 + (result[1]-targ[1])**2 )
                print("prediction {} for {}, pixel L2 {:.1f}".format(result, idx, L2))
                L2_results.append(L2)
        
        print("L2s: {:.1f} +/- {:.1f}".format(np.mean(L2_results), np.std(L2_results)))


if __name__ == "__main__":
    test = Test()
    path = '/nfs/diskstation/seita/bed-make/cache_white_v01/'
    all_cv_files = [join(path,x) for x in os.listdir(path) if 'cv_' in x]
    test.test(all_cv_files)
