"""Use this script to check H's datasets.

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
# Note: for h_v03         (i.e., dataset_2) they saved as one-item (or zero-item) lists.
# Note: for h_v03_success (i.e., dataset_3) they reverted back to just one item (i.e., dict)
# Note: for h_v04         (i.e., dataset_4) hopefully same, just one dict per file ...
# Note: for h_v04_success (i.e., dataset_5)
ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts_h_v04_dataset_4/' # change `dataset` later
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_h_v04_dataset_4/' # change `dataset` later
dataset  = 4 # change this!
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
files = sorted([x for x in os.listdir(ROLLOUTS)])

# --- CUTOFF, EXTREMELY IMPORTANT (in meters if images are from the Fetch) ---
ROBOT = 'Fetch'

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


total_class0 = 0
total_class1 = 0


for ff in files:
    # They saved in format: `rollout_TOP_Grasp_K/rollout.pkl` where `number=K`.
    # Um, well not in dataset_4 ... which is the grasp data from their actual rollouts.
    pkl_f = os.path.join(ff,'rollout.pkl')
    number = str(ff.split('_')[-1])
    file_path = os.path.join(ROLLOUTS,pkl_f)
    data = pickle.load(open(file_path,'rb'))

    # Debug and accumulate statistics for plotting later.
    print("\nOn data: {}, number {}, type {}".format(file_path, number, type(data)))

    # Change from their dataset_1 (I call cache_h_v01) to dataset_2 (I call cache_h_v03).
    # Because they made things length-1 lists ...
    if len(data) == 0:
        print("length is 0, skipping...")
        continue

    # The len(data)==1 and idx 0 extraction is only needed for `dataset_2`.
    # A bunch of checks to ensure that things are making sense ...
    if dataset == 2:
        assert len(data) == 1
        data = data[0]
    print("keys: {}".format(data.keys()))
    print("    data['pose']:   {}".format(np.array(data['pose'])))
    print("    data['class']:  {}".format(np.array(data['class'])))
    print("    data['side']:   {}".format(np.array(data['side'])))
    print("    data['type']:   {}".format(np.array(data['type'])))
    if data['type'] == 'success':
        if data['class'] == 0:
            total_class0 += 1
        elif data['class'] == 1:
            total_class1 += 1
        else:
            raise ValueError()
    assert data['c_img'].shape == (480, 640, 3), data['c_img'].shape
    if dataset == 2 or dataset == 3 or dataset == 5:
        assert data['d_img'].shape == (480, 640), data['d_img'].shape
    if dataset == 4:
        assert data['d_img'].shape == (480, 640, 3), data['d_img'].shape

    # Deal with paths and load the images. Note, here need to patch NaNs. I did this
    # _before_ actually saving the rollouts, so I patched it at a different step.
    num = str(number).zfill(4)
    c_path = os.path.join(IMG_PATH, 'num_{}_rgb.png'.format(num))
    d_path = os.path.join(IMG_PATH, 'num_{}_depth.png'.format(num))
    c_img = (data['c_img']).copy()
    d_img = (data['d_img']).copy()
    if dataset == 2 or dataset == 3 or dataset == 5:
        cv2.patchNaNs(d_img, 0) # I did this
        d_img = depth_to_net_dim(d_img, robot=ROBOT)
    assert d_img.shape == (480, 640, 3)
    pos = ( int(data['pose'][0]), int(data['pose'][1]) )

    # Can overlay images with the marker position, which presumably is the target.
    #cv2.circle(c_img, center=pos, radius=2, color=RED,   thickness=-1)
    #cv2.circle(c_img, center=pos, radius=3, color=BLACK, thickness=1)
    cv2.circle(d_img, center=pos, radius=2, color=RED,   thickness=-1)
    cv2.circle(d_img, center=pos, radius=3, color=BLACK, thickness=1)

    if data['type'] == 'success':
        cc = data['class']
        c_path = c_path.replace('.png','_success_{}.png'.format(cc))
        d_path = d_path.replace('.png','_success_{}.png'.format(cc))
    cv2.imwrite(c_path, c_img)
    cv2.imwrite(d_path, d_img)
    total += 1

print("total images:  {}".format(total))
print("total_class0:  {}".format(total_class0))
print("total_class1:  {}".format(total_class1))
print("see IMG_PATH:  {}".format(IMG_PATH))
