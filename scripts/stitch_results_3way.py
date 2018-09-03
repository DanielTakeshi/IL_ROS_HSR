"""Use this script for combining various results form various files.

Ideally have 3 subplots side by side, for the three different datasets.
"""
import argparse, cv2, os, pickle, sys, matplotlib, utils
from os.path import join
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from collections import defaultdict

# ------------------------------------------------------------------------------
# ADJUST. HH is the directory named like: 'grasp_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# Put figure in just the current/existing directory.
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/grasp/'

DATA_TO_LISTS = {
    'cache_d_v01': [
        'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_4000_cv_True',
        'grasp_4_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_4000_cv_True',
     ],
    'cache_h_v03': [
        'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_4000_cv_True',
        'grasp_4_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_4000_cv_True',
    ],
    'cache_combo_v01': [
        'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_8000_cv_True',
        'grasp_4_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_8000_cv_True',
    ],
}

DATA_TO_IDX = {
    'cache_d_v01': 0,
    'cache_h_v03': 1,
    'cache_combo_v01': 2,
}

NAMES = [
    'YOLO Pre-Trained',
    'Augmented AlexNet', # xavier + ReLU
]

# For the plot(s). There are a few plot-specific parameters, though.
# Also, sizes make more sense for each subplot being about (10 x 8).
tsize = 35
xsize = 30
ysize = 30
tick_size = 30
legend_size = 30
alpha = 0.5
error_alpha = 0.3
colors = ['blue', 'red', 'green']
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def subplots_pixels(ss_dict):
    """Make plot (w/subplots) of pixel losses only.

    Hopefully this will be in the final paper.
    Adjust the 'names' for legends. Look at `HH_LIST` to see what the labels should be.
    """
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, figsize=(11*ncols,9*nrows), squeeze=False)

    for d_idx,d_name in enumerate(ss_dict):
        ss_list = ss_dict[d_name]
        sp_idx = DATA_TO_IDX[d_name]
        print("plotting d_name {} at subplot idx {}".format(d_name, sp_idx))

        for i,name in enumerate(NAMES):
            ss = ss_list[i]
            all_train = np.array(ss['train'])
            all_test = np.array(ss['test'])
            all_test_raw = np.array(ss['raw_test'])
            all_lrates = np.array(ss['lrates'])
            epochs = ss['epoch']

            # {train, test, raw_test} = shape (K,N) where N is how often we recorded it.
            # For plots, ideally N = E, where E is number of epochs, but usually N > E.
            # For all cross validation folds, we must have N be the same.
            assert all_test.shape == all_test_raw.shape
            K, N = all_train.shape
            print("all_train.shape:  {}".format(all_train.shape))
            print("all_test.shape:   {}".format(all_test.shape))
            print("epoch: {}\n".format(epochs))

            # Since N != E in general, try to get epochs to line up.
            xs = np.arange(N) * (epochs[0] / float(N))
            mean_raw    = np.mean(all_test_raw, axis=0)
            std_raw     = np.std(all_test_raw, axis=0)
            mean_scaled = np.mean(all_test, axis=0)
            std_scaled  = np.std(all_test, axis=0)

            label_raw  = '{}; min {:.1f}'.format(name, np.min(mean_raw))
            ax[0,sp_idx].plot(xs, mean_raw, lw=2, color=colors[i], label=label_raw)
            ax[0,sp_idx].fill_between(xs, mean_raw-std_raw, mean_raw+std_raw,
                    alpha=error_alpha, facecolor=colors[i])

    # Bells and whistles
    for cc in range(ncols):
        ax[0,cc].set_ylim([0,100]) # TUNE !!
        ax[0,cc].set_xlabel('Training Epochs Over Augmented Data', fontsize=xsize)
        ax[0,cc].tick_params(axis='x', labelsize=tick_size)
        ax[0,cc].tick_params(axis='y', labelsize=tick_size)
        ax[0,cc].set_ylabel('Average Test L2 Loss (in Pixels)', fontsize=ysize)
        # Increase legend line size w/out affecting plot. :-)
        leg = ax[0,cc].legend(loc="best", ncol=1, prop={'size':legend_size})
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)

    # Hand-tuned titles!
    ax[0,0].set_title("Grasp Network, HSR Data", fontsize=tsize)
    ax[0,1].set_title("Grasp Network, Fetch Data", fontsize=tsize)
    ax[0,2].set_title("Grasp Network, Combined Data", fontsize=tsize)

    plt.tight_layout()
    figname = "fig_stitch_results_pixels_all_data_v03.png"
    plt.savefig(figname)
    print("Look at this figure:\n{}".format(figname))


if __name__ == "__main__":
    ss_dict = {}

    for data_name in DATA_TO_LISTS:
        print("data_name: {}".format(data_name))
        ss_list = []

        for tr in DATA_TO_LISTS[data_name]:
            path = join(HEAD, data_name, tr)
            pfiles = sorted([x for x in os.listdir(path) if '_raw_imgs.p' in x])
            print("  path: {}".format(path))
            ss = defaultdict(list) # for plotting later

            for cv_index,pf in enumerate(pfiles):
                # Now on one of the cross-validation splits (or a 'normal' training run).
                other_pf = pf.replace('_raw_imgs.p','.p')
                print("     --- Now on: {} ---".format(other_pf))
                data_other = pickle.load( open(join(path,other_pf),'rb') )
                y_pred = data_other['preds']
                y_targ = data_other['targs']
                assert len(y_pred) == len(y_targ)
                K = len(y_pred)

                # Later, figure out what to do if not using cross validation ...
                if 'cv_indices' in data_other:
                    cv_fname = data_other['cv_indices']
                    print("    processing CV file name: {}, idx {}, w/{} images".format(
                            cv_fname, cv_index, K))

                # `idx` = index into y_pred, y_targ, etc., _within_ this CV test set.
                for idx in range(K):
                    pred = y_pred[idx]
                    targ = y_targ[idx]
                    L2 = np.sqrt( (pred[0]-targ[0])**2 + (pred[1]-targ[1])**2)

                    # For plotting later. Note, `targ` is the ground truth.
                    ss['all_targs'].append(targ)
                    ss['all_preds'].append(pred)
                    ss['all_L2s'].append(L2)

                # Add some more stuff about this cv set, e.g., loss curves.
                ss['train'].append(data_other['train'])
                ss['test'].append(data_other['test'])
                ss['raw_test'].append(data_other['raw_test'])
                ss['epoch'].append(data_other['epoch'])
                ss['lrates'].append(data_other['lrates'])
            ss_list.append(ss)
        ss_dict[data_name] = ss_list

    print("\nDone with data loading. ss_dict.keys: {}".format(ss_dict.keys()))
    print("Now plot!\n")
    subplots_pixels(ss_dict)
