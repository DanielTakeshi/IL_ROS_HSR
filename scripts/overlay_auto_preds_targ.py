"""Use this script for making visuals with overlaid predictions and targets.

This automates the task, given some scripts that I ran, e.g. after:
    ./main/grasp.sh | tee logs/grasp.log
Better than copying/pasting, after all...

So, we need a bunch of those pickle files that we stored out of training.
They should have the best set of predictions, along with the actual labels.
From there, we have the information we need to do analysis and figures.

CAUTION! This assumes I did it using my fast data collection where there's
exactly two grasps (and two successes) per rollouts file. This will need to
be adjusted if that is not the case ...
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim

# ADJUST
ROLLOUT_HEAD = '/nfs/diskstation/seita/bed-make/rollouts_white_v01'
RESULTS_PATH = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/white_v01'
OUTPUT_PATH  = '/nfs/diskstation/seita/bed-make/figures/white_v01'
pfiles = sorted([x for x in os.listdir(RESULTS_PATH) if '_raw_imgs.p' in x])

INNER = 3
OUTER = 4


for pf in pfiles:
    other_pf = pf.replace('_raw_imgs.p','.p')
    print("\n\n\n ----- Now on: {}\n ----- with:   {}".format(pf, other_pf))
    data_imgs  = pickle.load( open(os.path.join(RESULTS_PATH,pf),'rb') )
    data_other = pickle.load( open(os.path.join(RESULTS_PATH,other_pf),'rb') )

    # Load predictions, targets, and images.
    y_pred = data_other['preds']
    y_targ = data_other['targs']
    c_imgs = data_imgs['c_imgs_list']
    d_imgs = data_imgs['d_imgs_list']
    assert len(y_pred) == len(y_targ) == len(c_imgs) == len(d_imgs)
    if 'cv_indices' in data_other:
        cv_ids = data_other['cv_indices']
    print("Now dealing with CV (rollout) indices: {}".format(cv_ids))

    # Index into y_pred, y_targ, c_imgs, d_imgs.
    idx = 0

    for rnum in cv_ids:
        print("\n=====================================================================")
        path = os.path.join(ROLLOUT_HEAD, 'rollout_{}/rollout.p'.format(rnum))
        if not os.path.exists(path):
            print("Error: {} does not exist".format(path))
            sys.exit()
        data = pickle.load(open(path,'rb'))
        print("loaded: {}".format(path))
        print("rollout {}, len(data): {}".format(rnum, len(data)))
        assert len(data) == 4
        assert data[0]['type'] == 'grasp'
        assert data[1]['type'] == 'success'
        assert data[2]['type'] == 'grasp'
        assert data[3]['type'] == 'success'

        # Unfortunately we'll assume that we have two grasps. For now we know this is the case, but
        # we can also load the rollout (as we do) and further inspect within that just to confirm.
        for g_in_rollout in range(2):
            c_path = os.path.join(OUTPUT_PATH, 'rollout_{}_grasp_{}_rgb.png'.format(rnum,g_in_rollout))
            d_path = os.path.join(OUTPUT_PATH, 'rollout_{}_grasp_{}_depth.png'.format(rnum,g_in_rollout))

            # Get these from our training run.
            pred = y_pred[idx]
            targ = y_targ[idx]
            cimg = c_imgs[idx].copy()
            dimg = d_imgs[idx].copy()
            idx += 1

            # Alternatively could get from rollout paths. Good to double check. Unfortunately again
            # this assumes I did grasp then success then grasp then success ... yeah.
            pose = data[g_in_rollout*2]['pose']
            print("pose: {}, targ: {} (should match)".format(pose, targ))
            pose = (int(pose[0]), int(pose[1]))
            targ = (int(targ[0]), int(targ[1]))
            #assert pose[0] == targ[0] and pose[1] == targ[1], "pose {}, targ {}".format(pose, targ)
            preds = (int(pred[0]), int(pred[1]))
            print("predictions: {}".format(preds))

            # Overlay the pose to the image (red circle, black border).
            cv2.circle(cimg, center=pose, radius=INNER, color=(0,0,255), thickness=-1)
            cv2.circle(dimg, center=pose, radius=INNER, color=(0,0,255), thickness=-1)
            cv2.circle(cimg, center=pose, radius=OUTER, color=(0,0,0), thickness=1)
            cv2.circle(dimg, center=pose, radius=OUTER, color=(0,0,0), thickness=1)
    
            # The PREDICTION, though, will be a large blue circle (yellow border?).
            cv2.circle(cimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
            cv2.circle(dimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
            cv2.circle(cimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)
            cv2.circle(dimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)
    
            cv2.imwrite(c_path, cimg)
            cv2.imwrite(d_path, dimg)
    
    print("=====================================================================")
print("\nDone, look at {}".format(OUTPUT_PATH))
