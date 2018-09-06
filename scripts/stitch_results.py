"""Use this script for combining various results form various files.

Ideally these are figures that can go in the final paper, which means there's always going to be
some manual fine-tuning anyway with legend labels and so forth.
"""
import argparse, cv2, os, pickle, sys, matplotlib, utils
import os.path as osp
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from collections import defaultdict

# ------------------------------------------------------------------------------
# ADJUST. HH is the directory named like: 'grasp_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# We'll just put the figure in the same directory this code is called.
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/'
DATA_NAME = 'cache_combo_v01'
HH_LIST = [
    'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_8000_cv_True',
    'grasp_4_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_8000_cv_True',
]

RESULTS_PATHS = []
RESULTS_LABELS = []
for HH  in HH_LIST:
    net_type = utils.net_check(HH)
    RESULTS_PATHS.append( osp.join(HEAD,net_type,DATA_NAME,HH) )
    RESULTS_LABELS.append( HH )

NAMES = [
    'YOLO Pre-Trained',
    'Augmented AlexNet', # xavier + ReLU
]
assert len(NAMES) == len(HH_LIST)

# For the plot(s). There are a few plot-specific parameters, though.
# Also, sizes make more sense for each subplot being about (10 x 8).
tsize = 30
xsize = 25
ysize = 25
tick_size = 25
legend_size = 25
alpha = 0.5
error_alpha = 0.3
colors = ['blue', 'red', 'green']

# Might as well make y-axis a bit more informative, if we know it.
LOSS_YLIMS = [
    [0.0, 0.050],
    None,
]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def make_plot_pixel_only(ss_list, with_min_value):
    """Make plot (w/subplots) of pixel losses only, hopefully for the final paper.

    Adjust the 'names' for legends. Look at `HH_LIST` to see the labels.
    This can go in an appendix to overlay performance of different methods.
    But in the main part of the paper, we might just do only one curve of
    the architecture we actually use.
    """
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(10*ncols,8*nrows), squeeze=False)

    for idx,(ss,name) in enumerate(zip(ss_list,NAMES)):
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
        print("on, name:         {}".format(name))
        print("all_train.shape:  {}".format(all_train.shape))
        print("all_test.shape:   {}".format(all_test.shape))
        print("epoch: {}\n".format(epochs))

        # Since N != E in general, try to get epochs to line up.
        xs = np.arange(N) * (epochs[0] / float(N))
        mean_raw    = np.mean(all_test_raw, axis=0)
        std_raw     = np.std(all_test_raw, axis=0)
        mean_scaled = np.mean(all_test, axis=0)
        std_scaled  = np.std(all_test, axis=0)

        label_raw  = '{}'.format(name)
        if with_min_value:
            label_raw  = '{}; min {:.1f}'.format(name, np.min(mean_raw))
        ax[0,0].plot(xs, mean_raw, lw=2, color=colors[idx], label=label_raw)
        ax[0,0].fill_between(xs, mean_raw-std_raw, mean_raw+std_raw,
                alpha=error_alpha, facecolor=colors[idx])

    # Bells and whistles
    #ax[0,0].set_xlim([0,14]) # TUNE !!
    ax[0,0].set_ylim([0,80]) # TUNE !!
    ax[0,0].set_xlabel('Training Epochs Over Augmented Data', fontsize=xsize)
    ax[0,0].tick_params(axis='x', labelsize=tick_size)
    ax[0,0].tick_params(axis='y', labelsize=tick_size)
    ax[0,0].set_ylabel('Average Test L2 Loss (Scaled)', fontsize=ysize)
    ax[0,0].set_ylabel('Average Test L2 Loss (in Pixels)', fontsize=ysize)
    ax[0,0].set_title("Grasp Point Cross-Validation Predictions", fontsize=tsize)
    leg = ax[0,0].legend(loc="best", ncol=1, prop={'size':legend_size})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)

    plt.tight_layout()
    suffix = "fig_stitch_results_pixels_{}_curves_v01.png".format(len(ss_list))
    figname = osp.join(suffix)
    plt.savefig(figname)
    print("Look at this figure:\n{}".format(figname))


def make_plot(ss_list):
    """Make plot (w/subplots) of losses, etc.

    First column: scaled. Second column: pixels.
    """
    nrows = 1
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(8*ncols,6*nrows), squeeze=False)

    for idx,(ss,name) in enumerate(zip(ss_list,RESULTS_LABELS)):
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
        print("on, name:         {}".format(name))
        print("all_train.shape:  {}".format(all_train.shape))
        print("all_test.shape:   {}".format(all_test.shape))
        print("epoch: {}\n".format(epochs))

        # Since N != E in general, try to get epochs to line up.
        xs = np.arange(N) * (epochs[0] / float(N))

        mean_raw    = np.mean(all_test_raw, axis=0)
        std_raw     = np.std(all_test_raw, axis=0)
        mean_scaled = np.mean(all_test, axis=0)
        std_scaled  = np.std(all_test, axis=0)

        label_scaled = '{}; minv_{:.4f}'.format(name[:7], np.min(mean_scaled))
        label_raw    = '{}; minv_{:.4f}'.format(name[:7], np.min(mean_raw))
        ax[0,0].plot(xs, mean_scaled, lw=2, color=colors[idx], label=label_scaled)
        ax[0,1].plot(xs, mean_raw,    lw=2, color=colors[idx], label=label_raw)
        ax[0,0].fill_between(xs,
                mean_scaled-std_scaled,
                mean_scaled+std_scaled,
                alpha=error_alpha,
                facecolor=colors[idx])
        ax[0,1].fill_between(xs,
                mean_raw-std_raw,
                mean_raw+std_raw,
                alpha=error_alpha,
                facecolor=colors[idx])

    # Bells and whistles
    for i in range(nrows):
        for j in range(ncols):
            if LOSS_YLIMS[idx] is not None:
                ax[i,j].set_ylim(LOSS_YLIMS[idx])
            ax[i,j].legend(loc="best", ncol=1, prop={'size':legend_size})
            ax[i,j].set_xlabel('Training Epochs (Over Augmented Data)', fontsize=xsize)
            ax[i,j].tick_params(axis='x', labelsize=tick_size)
            ax[i,j].tick_params(axis='y', labelsize=tick_size)
    ax[0,0].set_ylabel('Average Test L2 Loss (Scaled)', fontsize=ysize)
    ax[0,1].set_ylabel('Average Test L2 Loss (in Pixels)', fontsize=ysize)
    ax[0,0].set_title("Grasp Point Cross-Validation Predictions, Scaled", fontsize=tsize)
    ax[0,1].set_title("Grasp Point Cross-Validation Predictions", fontsize=tsize)

    plt.tight_layout()
    figname = osp.join("fig_stitch_results.png")
    plt.savefig(figname)
    print("Look at this figure:\n{}".format(figname))


if __name__ == "__main__":
    ss_list = []

    for RESULTS_PATH in RESULTS_PATHS:
        print("on RESULTS_PATH: {}".format(RESULTS_PATH))
        pfiles = sorted([x for x in os.listdir(RESULTS_PATH) if '_raw_imgs.p' in x])
        ss = defaultdict(list) # for plotting later

        for cv_index,pf in enumerate(pfiles):
            # Now on one of the cross-validation splits (or a 'normal' training run).
            other_pf = pf.replace('_raw_imgs.p','.p')
            print(" ----- Now on: {}\n ----- with:   {}".format(pf, other_pf))
            data_other = pickle.load( open(osp.join(RESULTS_PATH,other_pf),'rb') )
            y_pred = data_other['preds']
            y_targ = data_other['targs']
            assert len(y_pred) == len(y_targ)
            K = len(y_pred)

            # Later, figure out what to do if not using cross validation ...
            if 'cv_indices' in data_other:
                cv_fname = data_other['cv_indices']
                print("Now processing CV file name: {}, idx {}, with {} images".format(
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

    print("\nDone with data loading. Now making the plots ...")
    #make_plot(ss_list)

    # We can do this with the full ss_list.
    # OR with only the first item, which is what we actually use!!
    print("\nMaking a plot with all curves overlaid.")
    make_plot_pixel_only(ss_list, with_min_value=True)
    print("\nMaking a plot with only one curve (should be YOLO PreTrained).")
    make_plot_pixel_only( [ss_list[0]], with_min_value=True)
