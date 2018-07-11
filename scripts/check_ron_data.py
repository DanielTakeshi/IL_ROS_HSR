"""Use this script to check Ron's data.
"""
import cv2, pickle, sys, os
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim

ROLLOUTS = '/nfs/diskstation/seita/bed-make/blanket_corner_data/'
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_ron_data/'

files = sorted([x for x in os.listdir(ROLLOUTS) if x[-4:] == '.pkl'])

for pkl_f in files:
    number = str(pkl_f.split('.')[0])
    file_path = os.path.join(ROLLOUTS,pkl_f)
    data = pickle.load(open(file_path,'rb'))
    print("On data: {}, number {}".format(file_path, number))
    c_path = os.path.join(IMG_PATH,
            'ron_{}_rgb.png'.format(str(number).zfill(3))
    )
    d_path = os.path.join(IMG_PATH,
            'ron_{}_depth.png'.format(str(number).zfill(3))
    )
    assert data['RGBImage'].shape == (480, 640, 3)
    assert data['depthImage'].shape == (480, 640)
    c_img = (data['RGBImage']).copy()
    d_img = (data['depthImage']).copy()
    position = data['markerPos']

    ### UPDATE: normally, I'd do this;
    ### https://github.com/DanielTakeshi/fast_grasp_detect/blob/master/src/fast_grasp_detect/data_aug/depth_preprocess.py
    #d_img = depth_to_net_dim(data['depthImage'])
    #assert d_img.shape == (480, 640, 3)
    ### but for Ron's depth data, just use the single channel.

    # Overlay images with the marker position, which presumably is the target.
    pos = (position[0], position[1])
    cv2.circle(img=c_img, center=pos, radius=7, color=(0,0,255), thickness=-1)
    cv2.circle(img=d_img, center=pos, radius=7, color=(0,0,255), thickness=-1)
    cv2.circle(img=c_img, center=pos, radius=9, color=(0,0,0), thickness=2)
    cv2.circle(img=d_img, center=pos, radius=9, color=(0,0,0), thickness=2)

    cv2.imwrite(c_path, c_img)
    cv2.imwrite(d_path, d_img)
