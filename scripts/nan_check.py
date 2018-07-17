"""Use this script to inspect the raw data and check if NaNs exist in depth images."""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim

# Each key represents a data that maps to (ROLLOUTS, IMG_PATH) information.
datasets = {
    # My Cal data, (3->53). There's no held-out data. From using the 'slow' data collection script.
    'danielcal': (
        '/nfs/diskstation/seita/bed-make/rollouts/',
        '/nfs/diskstation/seita/bed-make/images_danielcal_data/'
    ),

    # Michael's blue data (not sure if original, 0-20 are weird?), (0->54) or (0->10) for held-out.
    'michaelblue_train': (
        '/nfs/diskstation/seita/laskey-data/bed_rcnn/rollouts/',
        '/nfs/diskstation/seita/bed-make/images_blue_data/'
    ),
    'michaelblue_valid': (
        '/nfs/diskstation/seita/laskey-data/bed_rcnn/held_out_bc/',
        '/nfs/diskstation/seita/bed-make/images_held_out_bc/'
    ),

    # Michael's blue data with DART applied, (0->57) or a different range for held-out.
    'michaelbluedart_train': (
        '/nfs/diskstation/seita/laskey-data/bed_rcnn/rollouts_dart/',
        '/nfs/diskstation/seita/bed-make/images_dart/'
    ),
    'michaelbluedart_valid': (
        '/nfs/diskstation/seita/laskey-data/bed_rcnn/held_out_dart/',
        '/nfs/diskstation/seita/bed-make/images_held_out_dart/'
    ),

    # Michael's NYTimes data. Note, I copied it so it's on _my_ data file.
    'michaelnytimes_train': (
        '/nfs/diskstation/seita/bed-make/rollouts_nytimes/',
        '/nfs/diskstation/seita/bed-make/images_nytimes/'
    ),
    'michaelnytimes_test': (
        '/nfs/diskstation/seita/bed-make/held_out_nytimes/',
        '/nfs/diskstation/seita/bed-make/images_held_out_nytimes/'
    ),
}

for key in datasets:
    ROLLOUTS, IMG_PATH = datasets[key]
    print("Now on key {}".format(key))
    print("ROLLOUTS: {}".format(ROLLOUTS))
    print("IMG_PATH: {}".format(IMG_PATH))

    for rnum in range(0, 60):
        print("\n=====================================================================")
        print("rollout {}".format(rnum))
        path = os.path.join(ROLLOUTS, 'rollout_{}/rollout.p'.format(rnum))
        if not os.path.exists(path):
            print("{} does not exist, skipping...".format(path))
            continue
        data = pickle.load(open(path,'rb'))

        for (d_idx,datum) in enumerate(data):
            # Ignore the first thing which is the 'starting' points (it exists for slow data collection).
            if type(datum) == list:
                continue
            print("currently on item {} in this rollout, out of {}:".format(d_idx,len(data)))
            # Some debug prints if we need for the future.
            #print('type:   {}'.format(datum['type']))
            #print('side:   {}'.format(datum['side']))
            #print('class:  {}'.format(datum['class']))
            #print('pose:   {}'.format(datum['pose']))

            # For some reason some datums dont't have depth images.
            if 'd_img' not in datum:
                print("no d_img in datum dict")
                continue

            # Check for NaN.
            assert not np.isnan(np.sum(datum['d_img']))

            ## All this does is modify the datum['d_img'] key; it leaves datum['c_img'] alone.
            #datum_to_net_dim(datum)

        print("=====================================================================")

print("All tests passed.")
