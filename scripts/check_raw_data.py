"""Use this script to inspect the raw data and record statistics, etc.

This is ideally the stuff that we can put in a paper appendix or supplementary website. Also, this
can be used to save the camera images so that we can later mix and match them for plots. For
instance, I can just save the 'success'-related images to show examples of success or failures.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)

# My Cal data, (3->53)
#ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts/'
#IMAGE_PATH = '/nfs/diskstation/seita/bed-make/images_danielcal_data/'

# Michael's blue data (not sure if original, rollouts 0-20 are weird?), (0->54) or (0->10) for held-out.
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/rollouts/'
#IMAGE_PATH = '/nfs/diskstation/seita/bed-make/images_blue_data/'
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/held_out_bc/'
#IMAGE_PATH = '/nfs/diskstation/seita/bed-make/images_held_out_bc/'

# Michael's blue data with DART applied, (0->57) or a different range for held-out.
ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/rollouts_dart/'
IMAGE_PATH = '/nfs/diskstation/seita/bed-make/images_blue_dart/'
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/held_out_dart/'
#IMAGE_PATH = '/nfs/diskstation/seita/bed-make/images_held_out_dart/'

g_total = 0
s_count_failure = 0
s_count_success = 0
rlengths = []

for rnum in range(0, 60):
    print("\n=====================================================================")
    print("rollout {}".format(rnum))
    path = os.path.join(ROLLOUTS, 'rollout_{}/rollout.p'.format(rnum))
    if not os.path.exists(path):
        print("{} does not exist, skipping...".format(path))
        continue
    data = pickle.load(open(path,'rb'))
    count = 0
    rlengths.append(len(data)-1)

    for (d_idx,datum) in enumerate(data):
        # Ignore the first thing which is the 'starting' points.
        if type(datum) == list:
            continue
        print("\ncurrently on item {} in this rollout, out of {}:".format(d_idx,len(data)))
        print('type:   {}'.format(datum['type']))
        print('side:   {}'.format(datum['side']))
        print('class:  {}'.format(datum['class']))
        print('pose:   {}'.format(datum['pose']))

        # Grasping
        if datum['type'] == 'grasp':
            pose = datum['pose']
            c_img = datum['c_img']
            img_path = os.path.join(IMAGE_PATH, 'rollout_{}_grasp_{}.png'.format(rnum,count))
            cv2.imwrite(img_path, c_img)
            g_total += 1

        # Success (0=success, 1=failure)
        elif datum['type'] == 'success':
            result = datum['class']
            if result == 0:
                s_count_success += 1
                img_path = os.path.join(IMAGE_PATH, 'rollout_{}_success_{}_class0.png'.format(rnum,count))
            else:
                s_count_failure += 1
                img_path = os.path.join(IMAGE_PATH, 'rollout_{}_success_{}_class1.png'.format(rnum,count))
            c_img = datum['c_img']
            cv2.imwrite(img_path, c_img)

            # Put here so increments only after (grasp,success) type pair.
            count += 1

        else:
            raise ValueError(datum['type'])

    print("=====================================================================")

print("\nSome stats (rollout length doesn't include the initial sampling point list:")
print("rollout lengths: {:.1f} +/- {:.1f} (sum: {})".format(np.mean(rlengths),np.std(rlengths),np.sum(rlengths)))
print("g_total:         {}".format(g_total)) # total images for grasping network
print("s_count_failure: {}".format(s_count_failure))
print("s_count_success: {}".format(s_count_success))
