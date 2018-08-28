"""Use this script to process as much of the SUCCESS data as possible.

See `convert_to_list_cache.py` for the grasping-based version.
This script is going to be similar, the main difference being that we can
borrow data from grasping network data to supplement FAILURE cases only.
This assumption that the grasping data = failure cases is important...

We want to have relatively equal class distribution, but also need there
to be a sufficient amount of 'borderline' cases.

As usual, need to shuffle and form CV splits beforehand. Again, result is
a list of dicts, each dict of which has cimg, dimg, class (all we need),
no data augmentation, and we just load this into our code.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from os.path import join
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from fast_grasp_detect.data_aug.data_augment import augment_data

# ----------------------------------------------------------------------------------------
# The usual rollouts path. Easiest if you make the cache with matching names.
# Right now it only supports `rollouts_d_vXY`, i.e., my data.,
DATA_PATH = '/nfs/diskstation/seita/bed-make/rollouts_d_v01/'
OUT_PATH  = '/nfs/diskstation/seita/bed-make/cache_d_v01_success/'
assert 'success' in OUT_PATH

# ALSO ADJUST, since we have slightly different ways of loading and storing data.
# This format depends on what we used for DATA_PATH, etc., in the above file names.
is_old_format = 'rollouts_white_v' in DATA_PATH
is_h_format   = 'rollouts_h_v' in DATA_PATH
is_d_format   = 'rollouts_d_v' in DATA_PATH
assert not (is_old_format and is_h_format and is_d_format)

# NOTE THE ROBOT, needed for the cutoff for depth images.
ROBOT = 'HSR'
# ----------------------------------------------------------------------------------------

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
NUM_CV = 10
INNER, OUTER = 3, 4

# List of dictionaries we'll split later for CV.
data_points = []
num_skipped = 0
num_successes = 0
num_failures = 0

if is_h_format:
    pass
elif is_old_format:
    pass
elif is_d_format:
    # --------------------------------------------------------------------------
    # My method, saved via 'fast' collection script. Has `t_success` and `b_success`.
    # The `files` should contain the _full_ path to the pickled data (each is a list).
    # Keys are 'class', 'c_img', 'd_img', and 'type', added for data augm code later.
    # FYI: this was with the HSR.
    # --------------------------------------------------------------------------
    G_DATA_PATH1 = join(DATA_PATH,'b_grasp')
    G_DATA_PATH2 = join(DATA_PATH,'t_grasp')
    S_DATA_PATH1 = join(DATA_PATH,'b_success')
    S_DATA_PATH2 = join(DATA_PATH,'t_success')

    # Iterate through all the success net and then a subset of the grasping data.
    files_grasping = sorted(
        [join(G_DATA_PATH1,x) for x in os.listdir(G_DATA_PATH1)] +
        [join(G_DATA_PATH2,x) for x in os.listdir(G_DATA_PATH2)]
    )
    files_success = sorted(
        [join(S_DATA_PATH1,x) for x in os.listdir(S_DATA_PATH1)] +
        [join(S_DATA_PATH2,x) for x in os.listdir(S_DATA_PATH2)]
    )

    for fidx,ff in enumerate(files_success):
        print("\n=====================================================================")
        data = pickle.load(open(ff,'rb'))
        print("On {}, len {}".format(ff,len(data)))

        for (d_idx,datum) in enumerate(data):
            assert 'c_img' in datum and 'd_img' in datum \
                    and 'class' in datum and 'type' in datum
            assert datum['type'] == 'success'
            assert datum['c_img'].shape == (480, 640, 3)
            assert datum['d_img'].shape == (480, 640)
            assert not np.isnan(np.sum(datum['d_img']))
            datum = datum_to_net_dim(datum, robot=ROBOT)
            assert datum['d_img'].shape == (480, 640, 3)
            if datum['class'] == 0:
                num_successes += 1
            elif datum['class'] == 1:
                num_failures += 1
            else:
                raise ValueError(datum['class'])
            data_points.append(datum)

    # Only do enough of the grasping data to get reasonable class balance.
    # Again, critically depends on assuming grasping data = failure cases.
    K = len(data_points)
    N_more = num_successes - num_failures
    print("\nFinished with collecting from success data. Class balance:")
    print("\tsuccesses: {} / {}".format(num_successes, K))
    print("\tfailures:  {} / {}".format(num_failures, K))
    print("Ideally want {} more grasping images".format(N_more))
    num_grasping = 0
    do_we_exit = False

    for fidx,ff in enumerate(files_grasping):
        if do_we_exit:
            break
        print("\n=====================================================================")
        data = pickle.load(open(ff,'rb'))
        print("On grasp data: {}, len {}".format(ff,len(data)))

        for (d_idx,datum) in enumerate(data):
            # Files spaced out. Got this idea from processing H's data.
            # In my case we skip here rather than the file index since I have a list here.
            if d_idx % 4 != 0:
                continue

            if datum['pose'][0] <= 56:
                print("    NOTE: not considering this due to pose: {}".format(datum['pose']))
                num_skipped += 1
                continue
            assert not np.isnan(np.sum(datum['d_img']))
            datum = datum_to_net_dim(datum, robot=ROBOT)
            assert datum['c_img'].shape == (480, 640, 3)
            assert datum['d_img'].shape == (480, 640, 3)
            assert 'c_img' in datum and 'd_img' in datum and 'pose' in datum
            datum['type'] = 'success'
            datum['class'] = 1 # i.e., a failure
            data_points.append(datum)
            num_grasping += 1
            num_failures += 1

            # Exit once we get a balance of failures and successes.
            if num_grasping >= N_more:
                do_we_exit = True
                break
    print("Now finished with succ/fail: {}, {}".format(num_successes, num_failures))
else:
    raise NotImplementedError()


# The indices which represent the CV split. NOTE THIS IS WHERE THE SHUFFLING HAPPENS!
# Well, OK, we shuffle for indices, but iterate through the `data_points` list in
# order of indices, so it's not 'entirely' random but we shuffle the training anyway.
N = len(data_points)
print("\nNow doing cross validation on {} points ...".format(N))
folds = [list(x) for x in np.array_split(np.random.permutation(N), NUM_CV) ]
for fold_indices in folds:
    print(fold_indices)

# These dictionaries do NOT have data augmentation applied!
for cv_idx, fold_indices in enumerate(folds):
    data_in_this_fold = [data_points[k] for k in range(N) if k in fold_indices]
    K = len(data_in_this_fold)
    out = os.path.join(OUT_PATH, 'success_list_of_dicts_nodaug_cv_{}_len_{}.pkl'.format(cv_idx,K))
    with open(out, 'w') as f:
        pickle.dump(data_in_this_fold, f)
    print("Just saved: {}, w/len {} data".format(out, K))
