"""Combines multiple datasets into the standardized data.

This assumes grasping data. E.g., we might combine our data and their data for grasping.
Those two datasets should have been processed via `convert_to_list_cache.py`. We will
have to 're-do' cross validation (and hence, shuffling).

Again, we should get a set of pickle files, each of which is a list, and the list items
are dicts, w/keys c_img, d_img, pose (i.e., the label), and type (needed for augmentation).

BUT ... since it may be interesting to see performance based on the particular dataset, we
will add that as another key, 'data_source'.

UPDATE Sept 05: use to get a smaller dataset. cache_combo_v02
UPDATE Sept 11: with new Honda data, use cache_combo_v03
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from os.path import join
from fast_grasp_detect.data_aug.depth_preprocess import (datum_to_net_dim, depth_to_net_dim)
from fast_grasp_detect.data_aug.data_augment import augment_data

# ----------------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/'
DATA_TO_COMBINE_NOHEAD = ['cache_d_v01', 'cache_h_v03',
                          'cache_h_v04_dataset_4',
                          'cache_h_v04_dataset_5']
DATA_TO_COMBINE = [join(HEAD, x) for x in DATA_TO_COMBINE_NOHEAD]
OUT_PATH = join(HEAD, 'cache_combo_v03')

for item in DATA_TO_COMBINE:
    assert 'cache_' in item
assert 'cache_' in OUT_PATH
assert 'success' not in OUT_PATH
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

NUM_CV = 10
FILES = []
for pth in DATA_TO_COMBINE:
    FILES.append( sorted([x for x in os.listdir(pth)]) )

IGNORE_FRAC = 0.10 # e.g., for `cache_combo_v02`, I used 0.75 to get a dataset 25% the size.
# ----------------------------------------------------------------------------------------

# List of dictionaries we'll split later for CV.
data_points = []

for fidx, (files, dhead, dtail) in enumerate(zip(FILES, DATA_TO_COMBINE, DATA_TO_COMBINE_NOHEAD)):
    print("Currently on {}-th set of files".format(fidx))
    print("the head directory: {}".format(dhead))
    print("and only the tail: {} (we use as another dict item)".format(dtail))
    for pkl_file in files:
        full_pkl_file = join(dhead, pkl_file)
        data = pickle.load(open(full_pkl_file,'rb'))
        print("    loaded: {} (has len {})".format(full_pkl_file, len(data)))
        for datum in data:
            assert 'c_img' in datum and 'd_img' in datum and 'type' in datum \
                    and 'pose' in datum, "here are keys: {}".format(datum.keys())
            datum['data_source'] = dtail
            if np.random.rand() > IGNORE_FRAC:
                data_points.append(datum)
        print("    finished this pickle file, len(data_points): {}".format(len(data_points)))

# ----------------------------------------------------------------------------------------
# The indices which represent the CV split. NOTE: THIS IS WHERE THE SHUFFLING HAPPENS!
# Well, techncially we still iterate through N so it's not 100% shuffling, but (a) for 
# training we shuffle anyway after augmentation, and (b) for test sets it doesn't matter.
# ----------------------------------------------------------------------------------------
N = len(data_points)
print("\nNow doing cross validation on {} points ...".format(N))
folds = [list(x) for x in np.array_split(np.random.permutation(N), NUM_CV) ]

# These dictionaries do NOT have data augmentation applied!
for cv_idx, fold_indices in enumerate(folds):
    data_in_this_fold = [data_points[k] for k in range(N) if k in fold_indices]
    K = len(data_in_this_fold)
    out = join(OUT_PATH, 'grasp_list_of_dicts_nodaug_cv_{}_len_{}.pkl'.format(cv_idx,K))
    with open(out, 'w') as f:
        pickle.dump(data_in_this_fold, f)
    print("Just saved: {}, w/len {} data".format(out, K))
