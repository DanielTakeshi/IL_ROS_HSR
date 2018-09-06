"""Use this script for inspecting results after we run.

ASSUMES WE SAVED DATA VIA CACHE, so we don't use `rollouts_X/rollouts_k/rollout.p`.

Update Aug 20, 2018: adding a third method here, `scatter_heat_final` which hopefully
can create a finalized version of a scatter plot figure that we might want to include.
The other figures created in this script here are mostly for quick debugging after a
training run.
"""
import argparse, cv2, os, pickle, sys, matplotlib, utils
import os.path as osp
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from collections import defaultdict

# ------------------------------------------------------------------------------
# ADJUST. HH is directory like: 'grasp_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/'
DATA_NAME = 'cache_combo_v01'
HH = 'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6000_cv_True'
DSOURCES = ['cache_d_v01', 'cache_h_v03']

# Sanity checks.
assert 'cache' in DATA_NAME
rgb_baseline = utils.rgb_baseline_check(HH)
net_type = utils.net_check(HH)
VIEW_TYPE = 'standard'
assert VIEW_TYPE in ['standard', 'close']

# Make directory. In separate `figures/`, we put the same directory name for results.
RESULTS_PATH = osp.join(HEAD, net_type, DATA_NAME, HH)
OUTPUT_PATH  = osp.join(HEAD, 'figures', DATA_NAME, HH)
if not osp.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Sizes for overlaying predictions/targets. Maybe cross hair is better?
INNER, OUTER = 3, 4

# For the plot(s). There are a few plot-specific parameters, though.
tsize = 36
xsize = 32
ysize = 32
tick_size = 32
legend_size = 32
alpha = 0.5
error_alpha = 0.3
error_fc = 'blue'
colors = ['blue', 'red', 'green']
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def scatter_heat_final(ss):
    """Finalized scatter plot for paper."""
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows), squeeze=False)
    xlim = [0, 640]
    ylim = [480, 0]
    VMIN = 0
    VMAX = 140

    # Put all stuff locally from dictionary input (turning into np.arrays as needed).
    all_targs = np.array(ss['all_targs'])
    all_preds = np.array(ss['all_preds'])
    all_L2s = np.array(ss['all_L2s'])
    all_x = ss['all_x']
    all_y = ss['all_y']
    all_names = ss['all_names']
    num_pts = len(ss['all_L2s'])

    # Heat-map of the L2 losses.
    if VIEW_TYPE == 'close':
        I = cv2.imread("scripts/imgs/image_example_close.png")
    elif VIEW_TYPE == 'standard':
        I = cv2.imread("scripts/imgs/daniel_data_example_file_15_idx_023_rgb.png")

    # Heat map now. To make color bars more equal:
    cf = ax[0,0].tricontourf(all_x, all_y, all_L2s, cmap='YlOrRd', vmin=VMIN, vmax=VMAX)
    cbar = fig.colorbar(cf, ax=ax[0,0])
    cbar.ax.tick_params(labelsize=40) 

    # Bells and whistles
    ax[0,0].imshow(I, alpha=alpha)
    ax[0,0].set_title("Pixel L2 Loss Heat Map",fontsize=tsize)
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].legend(loc="best", prop={'size':legend_size})
            ax[i,j].set_xlim(xlim)
            ax[i,j].set_ylim(ylim)
            ax[i,j].tick_params(axis='x', labelsize=tick_size)
            ax[i,j].tick_params(axis='y', labelsize=tick_size)

    plt.tight_layout()
    figname = "check_predictions_scatter_map_final_v01.png"
    plt.savefig(figname)
    print("Hopefully this figure can be in the paper:\n{}".format(figname))


def combine(ss):
    """Try to combine all together, if possible.

    THIS IS ALREADY REASONABLE, except that the color bar font size is bad,
    bleh.
    """
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows), squeeze=False)

    # Put all stuff locally from dictionary input (turning into np.arrays as needed).
    all_targs = np.array(ss['all_targs'])
    all_preds = np.array(ss['all_preds'])
    all_L2s = np.array(ss['all_L2s'])
    all_x = ss['all_x']
    all_y = ss['all_y']
    all_names = ss['all_names']
    num_pts = len(ss['all_L2s'])


    # --------------------------------------------
    # DO THE FIRST PLOT, the loss curve, mostly following `stitch_results.py`.
    all_train    = np.array(ss['train'])
    all_test     = np.array(ss['test'])
    all_test_raw = np.array(ss['raw_test'])
    all_lrates   = np.array(ss['lrates'])
    epochs       = ss['epoch']
    K, N         = all_train.shape
    xs           = np.arange(N) * (epochs[0] / float(N))
    mean_raw     = np.mean(all_test_raw, axis=0)
    std_raw      = np.std(all_test_raw, axis=0)
    mean_scaled  = np.mean(all_test, axis=0)
    std_scaled   = np.std(all_test, axis=0)

    name = 'YOLO Pre-Trained'
    idx = 0
    #label_raw  = '{}'.format(name)
    label_raw  = '{}; min {:.1f}'.format(name, np.min(mean_raw))
    ax[0,0].plot(xs, mean_raw, lw=2, color=colors[idx], label=label_raw)
    ax[0,0].fill_between(xs, mean_raw-std_raw, mean_raw+std_raw,
            alpha=error_alpha, facecolor=colors[idx])
    ax[0,0].set_ylim([0,80])
    ax[0,0].set_xlabel('Training Epochs Over Augmented Data', fontsize=xsize)
    ax[0,0].set_ylabel('Average Test L2 Loss (in Pixels)', fontsize=ysize)
    ax[0,0].set_title("Grasp Cross-Validation Predictions", fontsize=tsize)
    leg = ax[0,0].legend(loc="best", ncol=1, prop={'size':legend_size})
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    ax[0,0].tick_params(axis='x', labelsize=tick_size)
    ax[0,0].tick_params(axis='y', labelsize=tick_size)

    # --------------------------------------------
    # NOW DO THE SCATTER PLOT STUFF
    xlim = [0, 640]
    ylim = [480, 0]
    VMIN = 0
    VMAX = 140

    # Heat-map of the L2 losses.
    if VIEW_TYPE == 'close':
        I = cv2.imread("scripts/imgs/image_example_close.png")
    elif VIEW_TYPE == 'standard':
        I = cv2.imread("scripts/imgs/daniel_data_example_file_15_idx_023_rgb.png")

    # Create a scatter plot of where the targets are located.
    ax[0,1].imshow(I, alpha=alpha)
    ax[0,1].scatter(all_targs[:,0], all_targs[:,1], color='black')
    ax[0,1].set_title("Ground Truth ({} Points)".format(num_pts), fontsize=tsize)

    # Heat map now. To make color bars more equal:
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    # Better than before but still not perfect ... but I'll take it!
    # Also, for font sizes, use:
    # https://stackoverflow.com/questions/6567724/matplotlib-so-log-axis-only-has-minor-tick-mark-labels-at-specified-points-also/6568248#6568248

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    print(axes)
    print(divider)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax[0,2].imshow(I, alpha=alpha)
    cf = ax[0,2].tricontourf(all_x, all_y, all_L2s, cmap='YlOrRd', vmin=VMIN, vmax=VMAX)
    cbar = fig.colorbar(cf, ax=ax[0,2], cax=cax)
    cbar.ax.tick_params(labelsize=tick_size) 

    mean = np.mean(all_L2s)
    std = np.std(all_L2s)
    #ax[0,2].set_title("Pixel L2 Loss Heat Map: {:.1f} +/- {:.1f}".format(mean,std),fontsize=tsize)
    ax[0,2].set_title("Pixel L2 Loss Heat Map".format(mean,std),fontsize=tsize)

    # Bells and whistles
    for j in range(1,3):
        ax[0,j].legend(loc="best", prop={'size':legend_size})
        ax[0,j].set_xlim(xlim)
        ax[0,j].set_ylim(ylim)
        ax[0,j].tick_params(axis='x', labelsize=tick_size)
        ax[0,j].tick_params(axis='y', labelsize=tick_size)

    plt.tight_layout()
    figname = "check_predictions_scatter_map_final_v02.png"
    plt.savefig(figname)
    print("Hopefully this figure can be in the paper:\n{}".format(figname))


if __name__ == "__main__":
    print("RESULTS_PATH: {}".format(RESULTS_PATH))
    pfiles = sorted([x for x in os.listdir(RESULTS_PATH) if '_raw_imgs.p' in x])
    ss = defaultdict(list) # for plotting later

    for cv_index,pf in enumerate(pfiles):
        # Now on one of the cross-validation splits (or a 'normal' training run).
        other_pf = pf.replace('_raw_imgs.p','.p')
        print("\n\n ----- Now on: {}\n ----- with:   {}".format(pf, other_pf))
        data_imgs  = pickle.load( open(osp.join(RESULTS_PATH,pf),'rb') )
        data_other = pickle.load( open(osp.join(RESULTS_PATH,other_pf),'rb') )

        # Load predictions, targets, and images.
        y_pred = data_other['preds']
        y_targ = data_other['targs']
        c_imgs = data_imgs['c_imgs_list']
        d_imgs = data_imgs['d_imgs_list']
        assert len(y_pred) == len(y_targ) == len(c_imgs) == len(d_imgs)
        K = len(c_imgs)

        # Don't forget to split in case we use combo data.
        # We also need to track some more stuff when we go through results individually.
        if 'combo' in DATA_NAME:
            data_sources = data_other['test_data_sources']
            assert len(data_sources) == K
            ss['data_sources'].append(data_sources)

        # Later, figure out what to do if not using cross validation ...
        if 'cv_indices' in data_other:
            cv_fname = data_other['cv_indices']
            print("Now processing CV file name: {}, idx {}, with {} images".format(
                    cv_fname, cv_index, K))

        # `idx` = index into y_pred, y_targ, c_imgs, d_imgs, _within_ this CV test set.
        # Get these from training run, stored from best iteration on validation set.
        # Unlike with the non-cache case, we can't really load in rollouts easily due
        # to shuffling when forming the cache. We could get indices but not with it IMO.
        for idx in range(K):
            if idx % 10 == 0:
                print("  ... processing image {} in this test set".format(idx))
            pred = y_pred[idx]
            targ = y_targ[idx]
            cimg = c_imgs[idx].copy()
            dimg = d_imgs[idx].copy()
            L2 = np.sqrt( (pred[0]-targ[0])**2 + (pred[1]-targ[1])**2)

            # For plotting later. Note, `targ` is the ground truth.
            ss['all_targs'].append(targ)
            ss['all_preds'].append(pred)
            ss['all_x'].append(targ[0])
            ss['all_y'].append(targ[1])
            ss['all_L2s'].append(L2)
            targ  = (int(targ[0]), int(targ[1]))
            preds = (int(pred[0]), int(pred[1]))

        # Add some more stuff about this cv set, e.g., loss curves.
        ss['train'].append(data_other['train'])
        ss['test'].append(data_other['test'])
        ss['raw_test'].append(data_other['raw_test'])
        ss['epoch'].append(data_other['epoch'])
        ss['lrates'].append(data_other['lrates'])

        print("=====================================================================")

    scatter_heat_final(ss)
    combine(ss)
