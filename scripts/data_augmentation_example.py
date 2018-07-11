"""Use to inspect data augmentation.

Mostly mirrors `src/fast_grasp_detect/core/data_manager.py`.
Works for both RGB camera (10x images) and depth images (6x images).
The 10x and 6x include the original and also the flipped versions (about the y-axis).
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.data_augment import augment_data
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim

ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts/'
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_danielcal_data/'


def break_up_rollouts(rollout):
    """Break up the rollout as in the `fast_grasp_detect` code.
    `rollout`: odd-length list, starting point, then alternates between grasps and successes.
    `grasp_rollout`: list of lists, each of which are a `grasp_point`. All the list items are length
        one, so it's pointless to have this but perhaps there's backwards compatibility stuff.
    """
    grasp_point = []
    grasp_rollout = []
    for data in rollout:
        # As usual ignore the starting list
        if type(data) == list:
            continue
        if (data['type'] == 'grasp'):
            grasp_point.append(data)
        elif (data['type'] == 'success'):
            if len(grasp_point) > 0:
                grasp_rollout.append(grasp_point)
                grasp_point = []
    return grasp_rollout


def compute_label(datum):
    """Labels scaled in [-1,1], as described in paper. This is for _grasps_."""
    pose = datum['pose']
    label = np.zeros((2))
    x = pose[0]/640 - 0.5 # width: 640
    y = pose[1]/480 - 0.5 # height: 480
    label = np.array([x,y])
    return label


for rnum in range(3,53):
    ROLLOUT_NUM = rnum
    path = os.path.join(ROLLOUTS, 'rollout_{}/rollout.p'.format(ROLLOUT_NUM))
    if not os.path.exists(path):
        print("{} does not exist, skipping...".format(path))
        continue
    rollout = pickle.load(open(path,'rb'))
    grasp_rollout = break_up_rollouts(rollout)
    print("\n ****  loaded {}, len(grasp_rollout): {}  ****".format(path, len(grasp_rollout)))
    count = 0

    for grasp_point in grasp_rollout:
        print("\ninside grasp_point in grasp_rollout, length {}".format(len(grasp_point)))

        for data in grasp_point:
            # Adjust whether we want c_img or d_img. If d_img need to make it 3-channel.
            data_rgb = augment_data(data)
            datum_to_net_dim(data)
            data_depth = augment_data(data, depth_data=True)
            assert len(data_rgb) == 10 and len(data_depth) == 6 and \
                    type(data_rgb) is list and type(data_depth) is list

            # RGB
            # (If this were NN code, I'd run `datum_a['c_img']` through the YOLO network)
            # Save systematically along with original rollouts from other script
            for d_idx,datum_a in enumerate(data_rgb):
                label = compute_label(datum_a)
                img_suffix = 'rollout_{}_grasp_{}_dataaug_{}_RGB.png'.format(ROLLOUT_NUM, count, d_idx)
                img_path = os.path.join(IMG_PATH, img_suffix)
                cv2.imwrite(img_path, datum_a['c_img'])
                print("aug data idx {}, label {}, keys {}, saved {}".format(
                        d_idx, label, datum_a.keys(), img_suffix))
            # Depth.
            for d_idx,datum_a in enumerate(data_depth):
                label = compute_label(datum_a)
                img_suffix = 'rollout_{}_grasp_{}_dataaug_{}_DEPTH.png'.format(ROLLOUT_NUM, count, d_idx)
                img_path = os.path.join(IMG_PATH, img_suffix)
                cv2.imwrite(img_path, datum_a['c_img']) # 'c_img'; there's no 'd_img' key for aug.
                print("aug data idx {}, label {}, keys {}, saved {}".format(
                        d_idx, label, datum_a.keys(), img_suffix))
        count += 1
