"""Test the analytic grasping policy, and _perhaps_ analytic success net later.

But the focus here is really on the analytic _grasping_, as in the prior papers.
Use this to compute raw L2 losses in pixel space for the analytic method.

This should be run _right_after_ data collection. If results are good,
there is no need to use any Deep Learning.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
#from il_ros_hsr.p_pi.bed_making.analytic_success import Success_Net # requires hsrb_interface
from il_ros_hsr.p_pi.bed_making.analytic_grasp import Analytic_Grasp

# My Cal data, (3->53). There's no held-out data. From using the 'slow' data collection script.
#ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_danielcal_data/'

# Michael's blue data (not sure if original, 0-20 are weird?), (0->54) or (0->10) for held-out.
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/rollouts/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_blue_data/'
ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/held_out_bc/'
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_held_out_bc/'

# Michael's NYTimes data. Note, I copied it so it's on _my_ data file.
#ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts_nytimes/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_nytimes/'
#ROLLOUTS = '/nfs/diskstation/seita/bed-make/held_out_nytimes/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_held_out_nytimes/'


# Set up the baseline, etc. For colors, (0,0,0)=black, (0,0,255)=red, etc.
g_detector = Analytic_Grasp()
HEIGHT = 480
WIDTH = 640
DO_DEPTH = False
losses = []
gfactor = 3


for rnum in range(0, 60):
    print("\n=====================================================================")
    print("rollout {}".format(rnum))
    path = os.path.join(ROLLOUTS, 'rollout_{}/rollout.p'.format(rnum))
    if not os.path.exists(path):
        print("{} does not exist, skipping...".format(path))
        continue
    data = pickle.load(open(path,'rb'))

    # These record, for grasp and successes, the index into _this_ rollout.
    g_in_rollout = 0
    s_in_rollout = 0

    for (d_idx,datum) in enumerate(data):
        # Ignore the first thing which is the 'starting' points.
        if type(datum) == list:
            continue
        print("\ncurrently on item {} in this rollout, out of {}:".format(d_idx,len(data)))
        print('type:   {}'.format(datum['type']))
        print('side:   {}'.format(datum['side']))
        print('class:  {}'.format(datum['class']))
        print('pose:   {}'.format(datum['pose']))

        # All this does is modify the datum['d_img'] key; it leaves datum['c_img'] alone.
        datum_to_net_dim(datum)

        # Grasping. For these, overlay the actual pose to the image (red circle, black border).
        if datum['type'] == 'grasp':
            c_name = 'rollout_{}_gr_analytic_{}_rgb.png'.format(rnum, g_in_rollout)
            d_name = 'rollout_{}_gr_analytic_{}_depth.png'.format(rnum, g_in_rollout)
            c_path = os.path.join(IMG_PATH, c_name)
            d_path = os.path.join(IMG_PATH, d_name)
            c_img = (datum['c_img']).copy()
            d_img = (datum['d_img']).copy()

            # Predicted pose from baseline. See `main/deploy_analytic.py` for usage.
            c_img_small = cv2.resize(np.copy(c_img), (WIDTH/gfactor,HEIGHT/gfactor))
            c_data = g_detector.get_grasp(c_img_small, gfactor, fname=c_path)
            c_data *= gfactor
            c_pred = (int(c_data[0]), int(c_data[1]))
            cv2.circle(img=c_img, center=c_pred, radius=8, color=(255,0,0), thickness=-1)
            cv2.circle(img=c_img, center=c_pred, radius=10, color=(0,255,0), thickness=3)
            if DO_DEPTH:
                d_img_small = cv2.resize(np.copy(d_img), (WIDTH/gfactor,HEIGHT/gfactor))
                d_data = g_detector.get_grasp(d_img_small, gfactor, fname=d_path)
                d_data *= gfactor
                d_pred = (int(d_data[0]), int(d_data[1]))
                cv2.circle(img=d_img, center=d_pred, radius=8, color=(255,0,0), thickness=-1)
                cv2.circle(img=d_img, center=d_pred, radius=10, color=(0,255,0), thickness=3)

            # Now actual pose, then save with both prediction and target drawn.
            pose = datum['pose']
            pose_int = (int(pose[0]), int(pose[1]))
            cv2.circle(img=c_img, center=pose_int, radius=8, color=(0,0,255), thickness=-1)
            cv2.circle(img=c_img, center=pose_int, radius=10, color=(0,0,0), thickness=3)
            cv2.imwrite(c_path, c_img)
            if DO_DEPTH:
                cv2.circle(img=d_img, center=pose_int, radius=8, color=(0,0,255), thickness=-1)
                cv2.circle(img=d_img, center=pose_int, radius=10, color=(0,0,0), thickness=3)
                cv2.imwrite(d_path, d_img)
            g_in_rollout += 1

            # Also compute losses here to compare against learning-based techniques.
            predict = np.array([c_data[0], c_data[1]])
            target  = np.array([pose[0], pose[1]])
            loss = np.linalg.norm(predict-target)
            losses.append(loss)

        # Success (0=success, 1=failure). Skipping for now.
        elif datum['type'] == 'success':
            s_in_rollout += 1

        else:
            raise ValueError(datum['type'])

    print("=====================================================================")

print("\nRGB L2, raw pixel losses:")
print("avg: {:.2f} +/- {:.1f}".format(np.mean(losses), np.std(losses)))
print("max: {:.1f},  min: {:.1f}".format(np.max(losses), np.min(losses)))
