"""Use this script to inspect the raw data and record statistics, etc.

This is ideally the stuff that we can put in a paper appendix or supplementary website. Also, this
can be used to save the camera images so that we can later mix and match them for plots. For
instance, I can just save the 'success'-related images to show examples of success or failures.

Update: might as well save both the RGBs AND the depth.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim

# My Cal data, (3->53). There's no held-out data. From using the 'slow' data collection script.
#ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_danielcal_data/'

# Michael's blue data (not sure if original, 0-20 are weird?), (0->54) or (0->10) for held-out.
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/rollouts/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_blue_data/'
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/held_out_bc/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_held_out_bc/'

# Michael's blue data with DART applied, (0->57) or a different range for held-out.
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/rollouts_dart/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_dart/'
#ROLLOUTS = '/nfs/diskstation/seita/laskey-data/bed_rcnn/held_out_dart/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_held_out_dart/'

# Michael's NYTimes data. Note, I copied it so it's on _my_ data file.  He changed data so 
# we don't have annoying first element, but I think he has multiple TOP/BOTTOM cases?

# I need to investigate in more detail, he has some multiple grasp attempts but they are listed as
# successes? He also put the grasp stuff first, then the success stuff after, but that I am less
# concerned about as that's just re-shuffling data. UPDATE: got it, it's b/c of another script.
# TODO: now I need to check if the depth images make sense for rollouts 38+ ... and note that for
# Michael's faster data, he saves them by saving all the grasps first, THEN all the successes, hence
# why the order is different form mine, where I save the grasp, then success, then grasp, then
# success, etc., in order of when they were seen in the data. BTW, the good news is that with the
# NYTimes data, the initial list element (the one which is supposed to show the initial bed config)
# is not there, so the list lengths are EVEN numbers.

#ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts_nytimes/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_nytimes/'
#ROLLOUTS = '/nfs/diskstation/seita/bed-make/held_out_nytimes/'
#IMG_PATH = '/nfs/diskstation/seita/bed-make/images_held_out_nytimes/'

ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts_white_v01/'
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_white_v01/'

resize = False
g_total = 0
s_count_failure = 0
s_count_success = 0
rlengths = []

for rnum in range(0, 120):
    print("\n=====================================================================")
    print("rollout {}".format(rnum))
    path = os.path.join(ROLLOUTS, 'rollout_{}/rollout.p'.format(rnum))
    if not os.path.exists(path):
        print("{} does not exist, skipping...".format(path))
        continue
    data = pickle.load(open(path,'rb'))
    rlengths.append(len(data)-1)

    # These record, for grasp and successes, the index into _this_ rollout.
    g_in_rollout = 0
    s_in_rollout = 0

    for (d_idx,datum) in enumerate(data):
        # Ignore the first thing which is the 'starting' points.
        if type(datum) == list:
            continue
        print("\ncurrently on item {} in this rollout, out of {}:".format(d_idx,len(data)))
        print('type:   {}'.format(datum['type']))
        print('side:   {}'.format(datum['side']))
        if datum['type'] == 'grasp':
            print('pose:   {}'.format(datum['pose']))
        elif datum['type'] == 'success':
            print('class:  {}'.format(datum['class']))
        else:
            raise ValueError(datum['type'])

        # All this does is modify the datum['d_img'] key; it leaves datum['c_img'] alone.
        datum_to_net_dim(datum)

        # Resizing might be better to make input images smaller? (227,227,3) is AlexNet input.
        if resize:
            datum['c_img'] = cv2.resize(datum['c_img'], (227,227))
            datum['d_img'] = cv2.resize(datum['d_img'], (227,227))

        # Grasping. For these, overlay the pose to the image (red circle, black border).
        if datum['type'] == 'grasp':
            c_path = os.path.join(IMG_PATH, 'rollout_{}_grasp_{}_rgb.png'.format(rnum,g_in_rollout))
            d_path = os.path.join(IMG_PATH, 'rollout_{}_grasp_{}_depth.png'.format(rnum,g_in_rollout))
            c_img = (datum['c_img']).copy()
            d_img = (datum['d_img']).copy()
            pose = datum['pose']
            pose_int = (int(pose[0]), int(pose[1]))
            #if not resize:
            #    cv2.circle(img=c_img, center=pose_int, radius=4, color=(0,0,255), thickness=-1)
            #    cv2.circle(img=d_img, center=pose_int, radius=4, color=(0,0,255), thickness=-1)
            #    cv2.circle(img=c_img, center=pose_int, radius=5, color=(0,0,0), thickness=2)
            #    cv2.circle(img=d_img, center=pose_int, radius=5, color=(0,0,0), thickness=2)
            cv2.imwrite(c_path, c_img)
            cv2.imwrite(d_path, d_img)
            g_in_rollout += 1
            g_total += 1

        # Success (0=success, 1=failure).
        elif datum['type'] == 'success':
            result = datum['class']
            if result == 0:
                s_count_success += 1
                c_path = os.path.join(IMG_PATH,
                        'rollout_{}_success_{}_class0_rgb.png'.format(rnum,s_in_rollout)
                )
                d_path = os.path.join(IMG_PATH,
                        'rollout_{}_success_{}_class0_depth.png'.format(rnum,s_in_rollout)
                )
            else:
                s_count_failure += 1
                c_path = os.path.join(IMG_PATH,
                        'rollout_{}_success_{}_class1_rgb.png'.format(rnum,s_in_rollout)
                )
                d_path = os.path.join(IMG_PATH,
                        'rollout_{}_success_{}_class1_depth.png'.format(rnum,s_in_rollout)
                )
            cv2.imwrite(c_path, datum['c_img'])
            cv2.imwrite(d_path, datum['d_img'])
            s_in_rollout += 1

        else:
            raise ValueError(datum['type'])

    print("=====================================================================")

print("\nSome stats (rollout length doesn't include the initial sampling point list:")
print("rollout lengths: {:.1f} +/- {:.1f} (sum: {})".format(np.mean(rlengths),np.std(rlengths),np.sum(rlengths)))
print("g_total:         {}".format(g_total)) # total images for grasping network
print("s_count_failure: {}".format(s_count_failure))
print("s_count_success: {}".format(s_count_success))
