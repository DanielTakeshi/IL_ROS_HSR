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

    # https://github.com/DanielTakeshi/fast_grasp_detect/blob/master/src/fast_grasp_detect/data_aug/depth_preprocess.py
    d_img = depth_to_net_dim(data['depthImage'])
    assert d_img.shape == (480, 640, 3)

    cv2.imwrite(c_path, data['RGBImage'])
    cv2.imwrite(d_path, d_img)
