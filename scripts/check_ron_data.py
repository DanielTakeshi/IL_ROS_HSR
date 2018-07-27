"""Use this script to check Ron's data.

Run: `python scripts/check_ron_data.py`

Using the packages that I have in my Python 2.7 virtual environment, and assuming you
unzipped Ron's data into the correct rollotus path.

Ron tested with three different arm heights, and two different head and tilt values.
The data should probably be split to reflect that, and to be pre-processed so that it
fits in the format of our current data (so that we don't need to rewrite training code).

Also, for visualization I will plot Ron's data statistics.
"""
import cv2, pickle, sys, os
import numpy as np
np.set_printoptions(suppress=True, linewidth=200, precision=5)
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim

# --- Matplotlib stuff ---
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


# --- ADJUST PATHS ---
ROLLOUTS = '/nfs/diskstation/seita/bed-make/corner_ron_data_v02/'
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_ron_v02/'
files = sorted([x for x in os.listdir(ROLLOUTS) if x[-4:] == '.pkl'])
BLACK = (0,0,0)
RED = (0,0,255)


# For plotting later. I already know the bins of armState[0] and headState[0] by inspecting
# the data beforehand, btw.
as_0 = []
hs_0 = []
hs_1 = []
arm_bins = np.array([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
hs0_bins = np.array([-1.06465, -0.7854])
count = {}


def find_nearest(array, value):
    """For finding nearest bin value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def depth_scaled_to_255(img):
    img = 255.0/np.max(img)*img
    img = np.array(img,dtype=np.uint8)
    img = cv2.equalizeHist(img)
    return img


for pkl_f in files:
    number = str(pkl_f.split('.')[0])
    file_path = os.path.join(ROLLOUTS,pkl_f)
    data = pickle.load(open(file_path,'rb'))

    # Debug and accumulate statistics for plotting later.
    print("\nOn data: {}, number {}".format(file_path, number))
    print("    data['armState']:  {}".format(np.array(data['armState'])))
    print("    data['headState']: {}".format(np.array(data['headState'])))
    as_0.append(data['armState'][0])
    hs_0.append(data['headState'][0])
    hs_1.append(data['headState'][1])

    # For plotting, I'll add to path the appropriate bins for the arm and head state(s).
    arm_bin_idx = find_nearest(arm_bins, data['armState'][0])
    hs0_bin_idx = find_nearest(hs0_bins, data['headState'][0])
    key = 'a{}_h{}'.format(arm_bin_idx, hs0_bin_idx)
    if key in count:
        count[key] += 1
    else:
        count[key] = 1

    # Deal with paths and load the images.
    num = str(number).zfill(3)
    c_path = os.path.join(IMG_PATH,
            'ron_num{}_a{}_h{}_rgb.png'.format(num, arm_bin_idx, hs0_bin_idx)
    )
    d_path_1ch = os.path.join(IMG_PATH,
            'ron_num{}_a{}_h{}_depth_1ch.png'.format(num, arm_bin_idx, hs0_bin_idx)
    )
    assert data['RGBImage'].shape == (480, 640, 3)
    assert data['depthImage'].shape == (480, 640)
    c_img     = (data['RGBImage']).copy()
    d_img_1ch = (data['depthImage']).copy()
    position = data['markerPos']

    # I normally do this for preprocessing:
    # https://github.com/DanielTakeshi/fast_grasp_detect/blob/master/src/fast_grasp_detect/data_aug/depth_preprocess.py
    # Ron didn't do this but the above preprocessing makes the depth images much more human-interpretable.
    d_img_3ch = depth_to_net_dim(data['depthImage'])
    assert d_img_3ch.shape == (480, 640, 3)
    d_path_3ch = d_path_1ch.replace('1ch.png','3ch.png')

    # Overlay images with the marker position, which presumably is the target.
    # Update: probably don't need for the c_img, can leave off.
    pos = (position[0], position[1])
    #cv2.circle(c_img,     center=pos, radius=8,  color=RED,   thickness=-1)
    #cv2.circle(c_img,     center=pos, radius=10, color=BLACK, thickness=3)
    #cv2.circle(d_img_1ch, center=pos, radius=8,  color=RED,   thickness=-1)
    #cv2.circle(d_img_1ch, center=pos, radius=10, color=BLACK, thickness=3)
    #cv2.circle(d_img_3ch, center=pos, radius=8,  color=RED,   thickness=-1)
    #cv2.circle(d_img_3ch, center=pos, radius=10, color=BLACK, thickness=3)

    cv2.imwrite(c_path,     c_img)
    cv2.imwrite(d_path_1ch, d_img_1ch)
    cv2.imwrite(d_path_3ch, d_img_3ch)

    # Can try doing something similar for the single-channel images, making it easier to see.
    d_img_1ch_better = depth_scaled_to_255(d_img_1ch)
    cv2.imwrite(d_path_1ch.replace('1ch.png','1ch_b.png'), d_img_1ch_better)


# Finally, save some plots to investigate distribution of some data statistics.
nrows = 1
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols, 8*nrows),
    squeeze=False, sharex=False, sharey='col')

label0 = 'height_min_{:.5f}_max_{:.5f}'.format(np.min(as_0), np.max(as_0))
label1 = 'pan_min_{:.5f}_max_{:.5f}'.format(np.min(hs_0), np.max(hs_0))
label2 = 'tilt_min_{:.5f}_max_{:.5f}'.format(np.min(hs_1), np.max(hs_1))
axes[0,0].hist(as_0, bins=40, rwidth=0.95, label=label0)
axes[0,1].hist(hs_0, bins=40, rwidth=0.95, label=label1)
axes[0,2].hist(hs_1, bins=40, rwidth=0.95, label=label2)
axes[0,0].set_title('armState[0] (height?)', size=title_size)
axes[0,1].set_title('headState[0] (pan?)', size=title_size)
axes[0,2].set_title('headState[1] (tilt?)', size=title_size)

# Bells and whistles
for rr in range(nrows):
    for cc in range(ncols):
        axes[rr,cc].tick_params(axis='x', labelsize=tick_size)
        axes[rr,cc].tick_params(axis='y', labelsize=tick_size)
        axes[rr,cc].legend(loc="best", prop={'size':legend_size})
        axes[rr,cc].set_xlabel('Value in Ron\'s Data', size=xsize)
        axes[rr,cc].set_ylabel('Frequency in Data', size=ysize)
fig.tight_layout()
fpath = os.path.join(IMG_PATH, 'info.png')
fig.savefig(fpath)
print("\nJust saved data info plots at {}\n".format(fpath))


# Debug the counts to see the data combos.
for key in count:
    print("count[{}]: {}".format(key, count[key]))
