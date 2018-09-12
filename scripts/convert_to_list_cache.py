"""Use this script to process as much of the GRASPING data (not success) as possible.

Hopefully this will lead to faster data loading, which is becoming a huge limiting factor.
To be clear, we start with a `rollouts_X/` directory, from me or the other guys, and here we can
convert that to, say, `cache_X` which has arrays which we can load. We perform all steps up to but
not including data augmentation here, since that probably raises more hassles with splitting testing
vs training; testing data doesn't use data augmentation. We also deal with cross validation here.
Ideally, save 10 lists (if doing 10-fold) where each has a set of dictionaries.

And also we don't need to do any shuffling here, since the data manager will do that later.
EDIT: NO THIS IS NOT RIGHT, WE DO NEED TO SHUFFLE, THAT MUST BE DONE BEFORE CROSS-VALID SPLITS!!
But I do that anyway, I shuffle to get the CV splits where each index is one element ... whew!

After running this, I might have:

seita@autolab-titan-box:/nfs/diskstation/seita/bed-make$ ls -lh cache_white_v01/
total 1.1G
-rw-rw-r-- 1 nobody nogroup 120M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_0_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 122M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_1_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 121M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_2_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 119M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_3_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 124M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_4_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 121M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_5_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 121M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_6_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 120M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_7_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 119M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_8_len_21.pkl
-rw-rw-r-- 1 nobody nogroup 120M Aug 12 15:39 grasp_list_of_dicts_nodaug_cv_9_len_21.pkl

so, a list of dicts, each dict of which has c_img, d_img, pose (i.e., label), type, no data
augmentation, and we just load this into our code. _Should_ be easy ...
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from os.path import join
from fast_grasp_detect.data_aug.depth_preprocess import (datum_to_net_dim, depth_to_net_dim)
from fast_grasp_detect.data_aug.data_augment import augment_data

# ----------------------------------------------------------------------------------------
# The usual rollouts path. Easiest if you make the cache with matching names.
# Update: if using `rollouts_d_vXY` which is my data, have two sub-directories.
ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts_h_v04_dataset_5/'
OUT_PATH = '/nfs/diskstation/seita/bed-make/cache_h_v04_dataset_5/'
assert 'success' not in OUT_PATH

# ALSO ADJUST, since we have slightly different ways of loading and storing data.
# This format depends on what we used for ROLLOUTS, etc., in the above file names.
is_old_format = 'rollouts_white_v' in ROLLOUTS
is_h_format   = 'rollouts_h_v' in ROLLOUTS
is_d_format   = 'rollouts_d_v' in ROLLOUTS
assert not (is_old_format and is_h_format and is_d_format)
# ----------------------------------------------------------------------------------------

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
NUM_CV = 10
INNER, OUTER = 3, 4
files = sorted([x for x in os.listdir(ROLLOUTS)])

# List of dictionaries we'll split later for CV.
data_points = []
num_skipped = 0


if is_h_format:
    # --------------------------------------------------------------------------
    # They saved in format: `rollout_TOP_Grasp_K/rollout.pkl` where `number=K`.
    # Currently, each data point has keys:
    # ['headState', 'c_img', 'pose', 'class', 'd_img', 'type', 'armState', 'side']
    # This was with the Fetch.
    # --------------------------------------------------------------------------
    # UPDATE Sept 11: ugh, need to consider dataset_4, dataset_5, gets a bit hand-tuned ...
    # In particular, depth images are ALREADY processed in dataset_4, but not in dataset_5.
    # --------------------------------------------------------------------------
    for fidx,ff in enumerate(files):
        print("\n=====================================================================")

        pkl_f = os.path.join(ff,'rollout.pkl')
        number = str(ff.split('_')[-1])
        file_path = os.path.join(ROLLOUTS,pkl_f)
        datum = pickle.load(open(file_path,'rb'))
        assert type(datum) is dict

        # Debug and accumulate statistics for plotting later.
        print("\nOn data: {}, number {}".format(file_path, number))
        print("    data['pose']: {}".format(np.array(datum['pose'])))
        print("    data['type']: {}".format(np.array(datum['type'])))
        num = str(number).zfill(3)
        assert datum['c_img'].shape == (480, 640, 3)
        assert datum['d_img'].shape == (480, 640)
        cv2.patchNaNs(datum['d_img'], 0) # NOTE the patching!

        # As usual, datum to net dim must be done before data augmentation.
        datum = datum_to_net_dim(datum, robot='Fetch') # NOTE robot!
        assert datum['d_img'].shape == (480, 640, 3)
        assert 'c_img' in datum.keys() and 'd_img' in datum.keys() and 'pose' in datum.keys()

        # SKIP points that have pose[0] (i.e., x value) less than some threshold.
        if datum['pose'][0] <= 30:
            print("    NOTE: skipping this due to pose: {}".format(datum['pose']))
            num_skipped += 1
            continue
        if datum['type'] != 'grasp':
            print("    NOTE: skipping, type: {}".format(datum['type']))
            num_skipped += 1
            continue

        assert datum['type'] == 'grasp'
        data_points.append(datum)

elif is_old_format:
    # I saved as: `rollout_K/rollout.p`.
    # If it's in my format, I have to deal with multiple data points within a rollout,
    # distinguishing between grasping and successes (here, we care about grasping), etc.
    # Currently, each data point has keys:
    # ['perc', 'style', 'c_img', 'pose', 'd_img', 'type', 'side']
    # This was with the Fetch.
    # --------------------------------------------------------------------------
    for fidx,ff in enumerate(files):
        print("\n=====================================================================")

        pkl_f = os.path.join(ff,'rollout.p')
        number = str(ff.split('_')[-1])
        file_path = os.path.join(ROLLOUTS,pkl_f)
        data = pickle.load(open(file_path,'rb'))
        print("\nOn data: {}, number {}, has length {}".format(file_path, number, len(data)))

        # Unlike H case, here our 'data' is a list and we really want 'datum's.
        for (d_idx,datum) in enumerate(data):
            if type(datum) == list:
                continue
            if datum['type'] == 'success':
                continue
            assert datum['type'] == 'grasp'
            print("  on item {} in list:".format(d_idx))
            print('type: {}'.format(datum['type']))
            print('side: {}'.format(datum['side']))
            print('pose: {}'.format(datum['pose']))

            # All this does is modify the datum['d_img'] key; it leaves datum['c_img'] alone.
            # This will fail if there are NaNs, but I already patched beforehand.
            assert not np.isnan(np.sum(datum['d_img']))
            datum = datum_to_net_dim(datum, robot='HSR') # NOTE robot
            assert datum['c_img'].shape == (480, 640, 3)
            assert datum['d_img'].shape == (480, 640, 3)
            assert 'c_img' in datum.keys() and 'd_img' in datum.keys() and 'pose' in datum.keys()
            data_points.append(datum)

elif is_d_format:
    # My NEWER format, saved via 'fast' collection script. Has `t_grasp` and `b_grasp`.
    # Unlike the above, `files` actually contains the full pickle path to the data.
    # Only keys are 'pose', 'c_img', and 'd_img' but I need to add 'type' for downstream data
    # augmentation code.
    # FYI: This was with the HSR.
    # --------------------------------------------------------------------------
    ROLLOUTS1 = join(ROLLOUTS,'b_grasp')
    ROLLOUTS2 = join(ROLLOUTS,'t_grasp')
    files = sorted(
        [join(ROLLOUTS1,x) for x in os.listdir(ROLLOUTS1)] +
        [join(ROLLOUTS2,x) for x in os.listdir(ROLLOUTS2)]
    )

    for fidx,ff in enumerate(files):
        print("\n=====================================================================")
        data = pickle.load(open(ff,'rb'))
        print("On {}, len {}".format(ff,len(data)))

        for (d_idx,datum) in enumerate(data):
            print("  item {}, pose {}".format(d_idx, datum['pose']))
            assert not np.isnan(np.sum(datum['d_img']))
            datum = datum_to_net_dim(datum, robot='HSR') # NOTE robot
            assert datum['c_img'].shape == (480, 640, 3)
            assert datum['d_img'].shape == (480, 640, 3)
            assert 'c_img' in datum.keys() and 'd_img' in datum.keys() and 'pose' in datum.keys()
            # SKIP points that have pose[0] (i.e., x value) less than some threshold.
            if datum['pose'][0] <= 56:
                print("    NOTE: skipping this due to pose: {}".format(datum['pose']))
                num_skipped += 1
                continue
            assert 'type' not in datum
            datum['type'] = 'grasp'
            data_points.append(datum)

else:
    raise NotImplementedError()

print("Note: num_skipped: {}".format(num_skipped))


# The indices which represent the CV split.
# NOTE: THIS IS WHERE THE SHUFFLING HAPPENS!
N = len(data_points)
print("\nNow doing cross validation on {} points ...".format(N))
folds = [list(x) for x in np.array_split(np.random.permutation(N), NUM_CV) ]

# These dictionaries do NOT have data augmentation applied!
for cv_idx, fold_indices in enumerate(folds):
    data_in_this_fold = [data_points[k] for k in range(N) if k in fold_indices]
    K = len(data_in_this_fold)
    out = os.path.join(OUT_PATH, 'grasp_list_of_dicts_nodaug_cv_{}_len_{}.pkl'.format(cv_idx,K))
    with open(out, 'w') as f:
        pickle.dump(data_in_this_fold, f)
    print("Just saved: {}, w/len {} data".format(out, K))
