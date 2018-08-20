"""Use this script to check for my dataset.

I got these using `main/collect_data_bed_fast.py`.
This is where I can get lots of HSR-based data on my end.
"""
import cv2, pickle, sys, os
import numpy as np
from os.path import join
np.set_printoptions(suppress=True, linewidth=200, precision=5)
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim

# --- ADJUST PATHS ---
ROLLOUTS1 = '/nfs/diskstation/seita/bed-make/rollouts_d_v01/b_grasp'
ROLLOUTS2 = '/nfs/diskstation/seita/bed-make/rollouts_d_v01/t_grasp'
IMG_PATH  = '/nfs/diskstation/seita/bed-make/images_d_v01/'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
files = sorted(
    [join(ROLLOUTS1,x) for x in os.listdir(ROLLOUTS1)] +
    [join(ROLLOUTS2,x) for x in os.listdir(ROLLOUTS2)]
)
for f in files:
    print(f)

# --- Matplotlib ---
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

# cv2 and numbering
BLACK = (0,0,0)
RED = (0,0,255)
total = 0

# ------------------------------------------------------------------------------
all_x, all_y = [], []

for fidx,ff in enumerate(files):
    # I saved in format: `{b,t}_grasp/data.pkl`. The pickle files are _lists_.
    data_list = pickle.load(open(ff,'rb'))
    K = len(data_list)
    print("\nOn: {}, len {}".format(ff,K))
    total += K

    for didx,data in enumerate(data_list):
        # Debug and accumulate statistics for plotting later.
        print("    data['pose']: {}".format(np.array(data['pose'])))
        assert data['c_img'].shape == (480, 640, 3)
        assert data['d_img'].shape == (480, 640)

        # Deal with paths and load the images. Also, process depth images.
        num1 = str(fidx).zfill(2)
        num2 = str(didx).zfill(3)
        c_path = os.path.join(IMG_PATH, 'file_{}_idx_{}_rgb.png'.format(num1,num2))
        d_path = os.path.join(IMG_PATH, 'file_{}_idx_{}_depth.png'.format(num1,num2))
        c_img = (data['c_img']).copy()
        d_img = (data['d_img']).copy()

        #cv2.patchNaNs(d_img, 0) # Shouldn't be needed with HSR data.
        # NOTE NOTE NOTE!! This cutoff is HSR and situation dependent!
        d_img = depth_to_net_dim(d_img, cutoff=1400)
        assert d_img.shape == (480, 640, 3)

        pos = tuple(data['pose'])
        all_x.append(pos[0])
        all_y.append(pos[1])

        # Can overlay images with the marker position, which presumably is the target.
        cv2.circle(c_img, center=pos, radius=2, color=RED,   thickness=-1)
        cv2.circle(c_img, center=pos, radius=3, color=BLACK, thickness=1)
        cv2.circle(d_img, center=pos, radius=2, color=RED,   thickness=-1)
        cv2.circle(d_img, center=pos, radius=3, color=BLACK, thickness=1)
        cv2.imwrite(c_path, c_img)
        cv2.imwrite(d_path, d_img)
print("total images: {}".format(total))

# Let's get a scatter plot so we can see the distribution of data points.
I = cv2.imread('scripts/imgs/daniel_data_example_file_15_idx_023_rgb.png')
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.imshow(I, alpha=0.5)
ax.scatter(all_x, all_y, color='black')
ax.set_title('Scatter Plot of Data ({} Points)'.format(total), fontsize=title_size)
ax.set_xlim([0, 640])
ax.set_ylim([480, 0])
plt.tight_layout()
plt.savefig('scripts/imgs/daniel_data_scatter.png')
