"""Use this script to check other datasets.

v01: their pilot data set collection. Looks like all from type=grasp, side=TOP,
class=0, but I think that's a relic from earlier code and not to be interpreted.

Data keys: ['headState', 'c_img', 'pose', 'class', 'd_img', 'type', 'armState', 'side']

For the `depth_to_net_dim` method, see:
https://github.com/DanielTakeshi/fast_grasp_detect/blob/master/src/fast_grasp_detect/data_aug/depth_preprocess.py
and just copy and paste the relevant stuff here (no need to import the entire GitHub repo).
"""
import cv2, pickle, sys, os
import numpy as np
np.set_printoptions(suppress=True, linewidth=200, precision=5)
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim

# --- ADJUST PATHS ---
ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts_h_v01/'
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_h_v01/'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
files = sorted([x for x in os.listdir(ROLLOUTS)])

# --- Matplotlib and cv2 stuff ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
error_region_alpha = 0.25
title_size = 22
tick_size = 17
legend_size = 17
xsize = 18
ysize = 18
lw = 3
ms = 8
BLACK = (0,0,0)
RED = (0,0,255)
total = 0


for ff in files:
    # They saved in format: `rollout_TOP_Grasp_K/rollout.pkl` where `number=K`.
    pkl_f = os.path.join(ff,'rollout.pkl')
    number = str(ff.split('_')[-1])
    file_path = os.path.join(ROLLOUTS,pkl_f)
    data = pickle.load(open(file_path,'rb'))

    # Debug and accumulate statistics for plotting later.
    print("\nOn data: {}, number {}".format(file_path, number))
    print("    data['armState']:  {}".format(np.array(data['armState'])))
    print("    data['headState']: {}".format(np.array(data['headState'])))
    print("    data['pose']:      {}".format(np.array(data['pose'])))
    assert data['c_img'].shape == (480, 640, 3)
    assert data['d_img'].shape == (480, 640)

    # Deal with paths and load the images. Note, here need to patch NaNs. I did this
    # _before_ actually saving the rollouts, so I patched it at a different step.
    num = str(number).zfill(3)
    c_path = os.path.join(IMG_PATH, 'num_{}_rgb.png'.format(num))
    d_path = os.path.join(IMG_PATH, 'num_{}_depth.png'.format(num))
    c_img = (data['c_img']).copy()
    d_img = (data['d_img']).copy()
    cv2.patchNaNs(d_img, 0) # I did this
    d_img = depth_to_net_dim(d_img, cutoff=1.25)
    assert d_img.shape == (480, 640, 3)
    pos = tuple(data['pose'])

    # Can overlay images with the marker position, which presumably is the target.
    #cv2.circle(c_img, center=pos, radius=8,  color=RED,   thickness=-1)
    #cv2.circle(c_img, center=pos, radius=10, color=BLACK, thickness=3)
    #cv2.circle(d_img, center=pos, radius=8,  color=RED,   thickness=-1)
    #cv2.circle(d_img, center=pos, radius=10, color=BLACK, thickness=3)

    cv2.imwrite(c_path, c_img)
    cv2.imwrite(d_path, d_img)
    total += 1

print("total images: {}".format(total))
