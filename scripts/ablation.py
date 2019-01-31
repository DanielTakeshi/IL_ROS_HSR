"""Use this script for combining various results form various files.

Update: use this also for the ablation study with training on different sizes.
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
# ADJUST. HH is directory named like: 'grasp_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# We'll just put the figure in the same directory this code is called.
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/'
DATA_NAME = 'cache_combo_v03'
HH_LIST = [
    'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_9_to_9',
    'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_8_to_9',
    #'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_7_to_9',
    'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_6_to_9',
    #'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_5_to_9',
    'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_4_to_9',
    #'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_3_to_9',
    #'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_2_to_9',
    'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6001_cv_True_1_to_9',
]
NAMES = [
    '201 Training Images (~1/9)',
    '402 Training Images (~2/9)',
    #'604 Training Images (~3/9)',
    '805 Training Images (~4/9)',
    #'1007 Training Images (~5/9)',
    '1208 Training Images (~6/9)',
    #'1410 Training Images (~7/9)',
    #'1611 Training Images (~8/9)',
    '1813 Training Images (Full)',
]
RESULTS_PATHS = []
RESULTS_LABELS = []
for HH  in HH_LIST:
    net_type = utils.net_check(HH)
    RESULTS_PATHS.append( osp.join(HEAD,net_type,DATA_NAME,HH) )
    RESULTS_LABELS.append( HH )


assert len(NAMES) == len(HH_LIST)

# Oops I only record epoch, oh well, do NUM_STEPS instead ...
NUM_STEPS = 6000
BATCH_SIZE = 32

# For the plot(s). There are a few plot-specific parameters, though.
# Also, sizes make more sense for each subplot being about (10 x 8).
tsize = 30
xsize = 25
ysize = 25
tick_size = 25
legend_size = 25
alpha = 0.5
error_alpha = 0.3
colors = ['red', 'blue', 'gold', 'black', 'green', 'purple', 'gray']

# Might as well make y-axis a bit more informative, if we know it.
LOSS_YLIMS = [
    [0.0, 0.050],
    None,
]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def smooth(x, w=5):
    result = np.copy(x)
    for i in range(len(x)):
        result[i] = np.mean(x[i-w:i])
    return result


def make_plot(ss_list, with_min_value):
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
        # For all cross validation folds, we must have N be the same. Actually for
        # the ablation I just set K=1.
        assert all_test.shape == all_test_raw.shape
        K, N = all_train.shape
        print("on, name:         {}".format(name))
        print("all_train.shape:  {}".format(all_train.shape))
        print("all_test.shape:   {}".format(all_test.shape))
        print("epoch: {}\n".format(epochs))

        # For ablation, we know the number of steps. Take it and form the number
        # of total elements processed during training.
        xs = np.arange(N) * (NUM_STEPS / float(N)) * (BATCH_SIZE / 1000.0)
        mean_raw    = np.mean(all_test_raw, axis=0)
        std_raw     = np.std(all_test_raw, axis=0)
        mean_scaled = np.mean(all_test, axis=0)
        std_scaled  = np.std(all_test, axis=0)

        label_raw  = '{}'.format(name)
        if with_min_value:
            label_raw  = '{}; min {:.1f}'.format(name, np.min(mean_raw))
        ax[0,0].plot(xs, smooth(mean_raw), lw=2, color=colors[idx], label=label_raw)
        #ax[0,0].fill_between(xs, mean_raw-std_raw, mean_raw+std_raw,
        #        alpha=error_alpha, facecolor=colors[idx])

    # Bells and whistles
    #ax[0,0].set_xlim([0,14]) # TUNE !!
    ax[0,0].set_ylim([0,70]) # TUNE !!
    ax[0,0].set_xlabel('Training Points Consumed (Thousands)', fontsize=xsize)
    ax[0,0].tick_params(axis='x', labelsize=tick_size)
    ax[0,0].tick_params(axis='y', labelsize=tick_size)
    ax[0,0].set_ylabel('Average Test L2 Loss (in Pixels)', fontsize=ysize)
    ax[0,0].set_title("Predictions Based on Training Size", fontsize=tsize)
    leg = ax[0,0].legend(loc="best", ncol=1, prop={'size':legend_size})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)

    plt.tight_layout()
    suffix = "fig_ablation.png".format(len(ss_list))
    figname = osp.join(suffix)
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
    print("Note that len(ss_list): {}".format(len(ss_list)))
    make_plot(ss_list, with_min_value=False)

    ## # We can do this with the full ss_list.
    ## # OR with only the first item, which is what we actually use!!
    ## print("\nMaking a plot with all curves overlaid.")
    ## make_plot_pixel_only(ss_list, with_min_value=True)
    ## print("\nMaking a plot with only one curve (should be YOLO PreTrained).")
    ## make_plot_pixel_only( [ss_list[0]], with_min_value=True)
