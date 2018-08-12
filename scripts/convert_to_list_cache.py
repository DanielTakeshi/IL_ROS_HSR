"""Use this script to convert raw rollout data to numpy arrays.

Hopefully this will lead to faster data loading, which is becoming a huge limiting factor.

To be clear, we start with a `rollouts_X/` directory, from me or the other guys, and here we can
convert that to, say, `cache_X` which has arrays which we can load. We perform all steps up to but
not including data augmentation here, since that probably raises more hassles with splitting testing
vs training; testing data doesn't use data augmentation. We also deal with cross validation here.
Ideally, save 10 lists (if doing 10-fold) where each has a set of dictionaries.

And also we don't need to do any shuffling here, since the data manager will do that later.

Update: hmm ... right now we only use the grasping network's stuff.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from fast_grasp_detect.data_aug.data_augment import augment_data

# The usual rollouts path. Easiest if you make the cache with matching names.
ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts_white_v01/'
OUT_PATH = '/nfs/diskstation/seita/bed-make/cache_white_v01/'

# ALSO ADJUST, since we have slightly different ways of loading and storing data.
# This format depends on what we used for ROLLOUTS, etc., in the above file names.
is_my_format = True
is_h_format = False
assert not (is_my_format and is_h_format)

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
NUM_CV = 10
INNER, OUTER = 3, 4
files = sorted([x for x in os.listdir(ROLLOUTS)])

# List of dictionaries we'll split later for CV.
data_points = []


if is_h_format:
    # They saved in format: `rollout_TOP_Grasp_K/rollout.pkl` where `number=K`.
    # Currently, each data point has keys:
    # ['headState', 'c_img', 'pose', 'class', 'd_img', 'type', 'armState', 'side']
    for fidx,ff in enumerate(files):
        print("\n=====================================================================")

        pkl_f = os.path.join(ff,'rollout.pkl')
        number = str(ff.split('_')[-1])
        file_path = os.path.join(ROLLOUTS,pkl_f)
        datum = pickle.load(open(file_path,'rb'))

        # Debug and accumulate statistics for plotting later.
        print("\nOn data: {}, number {}".format(file_path, number))
        print("    data['pose']: {}".format(np.array(datum['pose'])))
        assert datum['c_img'].shape == (480, 640, 3)
        assert datum['d_img'].shape == (480, 640)
        num = str(number).zfill(3)
        cv2.patchNaNs(datum['d_img'], 0) # Note the patching!

        # As usual, datum to net dim must be done before data augmentation.
        datum = datum_to_net_dim(datum)
        assert datum['d_img'].shape == (480, 640, 3)
        assert 'c_img' in datum.keys() and 'd_img' in datum.keys() and 'pose' in datum.keys()
        data_points.append(datum)

elif is_my_format:
    # I saved as: `rollout_K/rollout.p`.
    # If it's in my format, I have to deal with multiple data points within a rollout,
    # distinguishing between grasping and successes (here, we care about grasping), etc.
    # Currently, each data point has keys:
    # ['perc', 'style', 'c_img', 'pose', 'd_img', 'type', 'side']
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
            datum = datum_to_net_dim(datum)
            assert datum['c_img'].shape == (480, 640, 3)
            assert datum['d_img'].shape == (480, 640, 3)
            assert 'c_img' in datum.keys() and 'd_img' in datum.keys() and 'pose' in datum.keys()
            data_points.append(datum)
else:
    raise NotImplementedError()


# The indices which represent the CV split.
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
