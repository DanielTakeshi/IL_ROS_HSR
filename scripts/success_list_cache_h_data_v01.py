"""Use this script to process as much of the SUCCESS data as possible.

For the H data, that is.

NOTE: they stored with opposite class labels, so just reverse it for mine.
And also I'd use their grasping network as success data, always.
BUT: sub-sample, since they have some temporally-correlated data, moreso than mine.
UPDATE: no, don't do that!

UPDATE UPDATE UPDATE: this is for

        rollout_h_v04_dataset_4.

Do NOT do dataset_5 here!!
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from os.path import join
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from fast_grasp_detect.data_aug.data_augment import augment_data

# ----------------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/'
DATA_PATH = join(HEAD, 'rollouts_h_v04_dataset_4')
OUT_PATH  = join(HEAD, 'cache_h_v04_dataset_4_success/')
assert 'success' in OUT_PATH
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
NUM_CV = 10

# NOTE THE ROBOT, needed for the cutoff for depth images.
ROBOT = 'Fetch'
# ----------------------------------------------------------------------------------------

# List of dictionaries we'll split later for CV.
data_points = []
num_skipped = 0
num_successes = 0
num_failures = 0

# ----------------------------------------------
# IN THEIR DATA, should be class=1 means success
# ----------------------------------------------
rollouts = [x for x in os.listdir( DATA_PATH ) if 'rollout' in x]

for fidx,ff in enumerate(rollouts):
    print("\n=====================================================================")
    subdirs = [x for x in os.listdir(join(DATA_PATH,ff)) if 'rollout' in x]

    assert 'rollout_0' in ff, ff
    assert 'rollout.pkl' in subdirs[0], subdirs[0]
    assert len(subdirs) == 1, subdirs

    fname = join(DATA_PATH, ff, subdirs[0])
    datum = pickle.load(open(fname,'rb'))
    print("On {}, len {}".format(fname, len(datum)))
    print("keys: {}".format(datum.keys()))
    print("datum['class']:  {} (_their_ format, 1=success)".format(datum['class']))
    print("datum['type']:   {}".format(datum['type']))

    assert 'c_img' in datum and 'd_img' in datum \
        and 'class' in datum and 'type' in datum
    assert datum['c_img'].shape == (480, 640, 3)
    assert datum['d_img'].shape == (480, 640, 3)
    assert not np.isnan(np.sum(datum['d_img']))

    if datum['type'] == 'success':
        assert datum['class'] == 1, datum['class']
        num_successes += 1
        datum['class'] = 0 # change to _my_ style
    elif datum['type'] == 'grasp':
        assert datum['class'] == 0, datum['class']
        # let's skip every few for better balance
        if fidx % 2 == 0:
            continue
        datum['class'] = 1 # change to _my_ style
        num_failures += 1

    data_points.append(datum)

K = len(data_points)
N_more = num_successes - num_failures
print("\n\n\nFinished with collecting from success data. Class balance:")
print("\tsuccesses: {} / {}".format(num_successes, K))
print("\tfailures:  {} / {}".format(num_failures, K))
print("Ideally want {} more grasping images, roughly spaced out\n\n".format(N_more))
num_grasping = 0
do_we_exit = False


print("\n\nNow finished with succ/fail: {}, {}\n".format(num_successes, num_failures))

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
