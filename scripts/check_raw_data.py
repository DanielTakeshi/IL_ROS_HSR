"""Use this script to inspect the raw data and record statistics, etc.

This is ideally the stuff that we can put in a paper appendix or supplementary website. Also, this
can be used to save the camera images so that we can later mix and match them for plots.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)

ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts/'
IMAGE_PATH = '/nfs/diskstation/seita/bed-make/images/'


for rnum in range(3, 53):
    print("\n=====================================================================")
    print("=====================================================================")
    print("rollout {}".format(rnum))
    path = os.path.join(ROLLOUTS, 'rollout_{}/rollout.p'.format(rnum))
    data = pickle.load(open(path,'rb'))
    count = 0

    for (d_idx,datum) in enumerate(data):
        # Ignore the first thing which is the 'starting' points.
        if type(datum) == list:
            continue
        print("\ncurrently on item {} in this rollout, out of {}:".format(d_idx, len(data)))
        print('type:   {}'.format(datum['type']))
        print('side:   {}'.format(datum['side']))
        print('class:  {}'.format(datum['class']))
        print('pose:   {}'.format(datum['pose']))

        # Grasping
        if datum['type'] == 'grasp':
            pose = datum['pose']
            c_img = datum['c_img']
            #img_path = os.path.join(IMAGE_PATH, 'rollout_{}_grasp_{}.png'.format(rnum,count))
            #cv2.imwrite(img_path, c_img)

        # Success
        elif datum['type'] == 'success':
            pass

        else:
            raise ValueError(datum['type'])

        count += 1

    print("=====================================================================")
    print("=====================================================================")
