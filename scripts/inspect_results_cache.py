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
tsize = 30
xsize = 25
ysize = 25
tick_size = 25
legend_size = 25
alpha = 0.5
error_alpha = 0.3
error_fc = 'blue'

# Might as well make y-axis a bit more informative, if we know it.
#LOSS_YLIM = [0.0, 0.035]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def scatter_heat_final(ss):
    """Finalized scatter plot for paper."""
    nrows, ncols = 1, 2
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
    # skipping over some of the debugging messages from the other method ...

    # Heat-map of the L2 losses.
    if VIEW_TYPE == 'close':
        I = cv2.imread("scripts/imgs/image_example_close.png")
    elif VIEW_TYPE == 'standard':
        I = cv2.imread("scripts/imgs/daniel_data_example_file_15_idx_023_rgb.png")
        #I = cv2.imread("scripts/imgs/example_image_h_v03.png")

    # Create a scatter plot of where the targets are located.
    ax[0,0].imshow(I, alpha=alpha)
    ax[0,0].scatter(all_targs[:,0], all_targs[:,1], color='black')
    ax[0,0].set_title("Ground Truth ({} Points)".format(num_pts), fontsize=tsize)

    # Heat map now. To make color bars more equal:
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    # Better than before but still not perfect ... but I'll take it!
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    print(axes)
    print(divider)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax[0,1].imshow(I, alpha=alpha)
    cf = ax[0,1].tricontourf(all_x, all_y, all_L2s, cmap='YlOrRd', vmin=VMIN, vmax=VMAX)
    fig.colorbar(cf, ax=ax[0,1], cax=cax)
    mean = np.mean(all_L2s)
    std = np.std(all_L2s)
    ax[0,1].set_title("Pixel L2 Loss Heat Map: {:.1f} +/- {:.1f}".format(mean, std), fontsize=tsize)

    # Bells and whistles
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].legend(loc="best", prop={'size':legend_size})
            ax[i,j].set_xlim(xlim)
            ax[i,j].set_ylim(ylim)
            ax[i,j].tick_params(axis='x', labelsize=tick_size)
            ax[i,j].tick_params(axis='y', labelsize=tick_size)

    plt.tight_layout()
    figname = osp.join(OUTPUT_PATH,"check_predictions_scatter_map_final.png")
    plt.savefig(figname)
    print("Hopefully this figure can be in the paper:\n{}".format(figname))


def make_scatter(ss):
    """Make giant scatter plot image."""
    nrows, ncols = 2, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(10*ncols,8*nrows))
    xlim = [0, 640]
    ylim = [480, 0]
    VMIN = 0
    VMAX = 200

    # Put all stuff locally from dictionary input (turning into np.arrays as needed).
    all_targs = np.array(ss['all_targs'])
    all_preds = np.array(ss['all_preds'])
    all_L2s = np.array(ss['all_L2s'])
    all_x = ss['all_x']
    all_y = ss['all_y']
    all_names = ss['all_names']

    num_pts = len(ss['all_L2s'])
    print("\nall_targs.shape: {}".format(all_targs.shape))
    print("all_preds.shape: {}".format(all_preds.shape))
    print("all_L2s.shape:   {}".format(all_L2s.shape))
    print("all_L2s (pixels): {:.1f} +/- {:.1f}".format(np.mean(all_L2s), np.std(all_L2s)))

    # Print names of the images with highest L2 errors for inspection later.
    print("\nHere are file names with highest L2 errors.")
    indices = np.argsort(all_L2s)[::-1]
    edge = 10
    for i in range(edge):
        print("{}".format(all_names[indices[i]]))
    print("...")
    for i in range(num_pts-edge, num_pts):
        print("{}".format(all_names[indices[i]]))

    # ------------------------------------------------------------------------------
    # Heat-map of the L2 losses. I was going to use the contour code but that
    # requires a value exists for every (x,y) pixel pair from our discretization.
    # To get background image in matplotlib:
    # https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image
    # This might be helpful:
    # https://stackoverflow.com/questions/48472227/how-can-one-create-a-heatmap-from-a-2d-scatterplot-data-in-python
    # ------------------------------------------------------------------------------
    if VIEW_TYPE == 'close':
        I = cv2.imread("scripts/imgs/image_example_close.png")
    elif VIEW_TYPE == 'standard':
        I = cv2.imread("scripts/imgs/daniel_data_example_file_15_idx_023_rgb.png")
        #I = cv2.imread("scripts/imgs/example_image_h_v03.png")

    # Create a scatter plot of where the targets are located.
    ax[0,0].imshow(I, alpha=alpha)
    ax[0,0].scatter(all_targs[:,0], all_targs[:,1], color='black')
    ax[0,0].set_title("Ground Truth ({} Points)".format(num_pts), fontsize=tsize)

    # Along with predictions.
    ax[0,1].imshow(I, alpha=alpha)
    ax[0,1].scatter(all_preds[:,0], all_preds[:,1], color='black')
    ax[0,1].set_title("Grasp Network Predictions (10-Fold CV)", fontsize=tsize)

    # Heat map now.
    ax[1,0].imshow(I, alpha=alpha)
    cf = ax[1,0].tricontourf(all_x, all_y, all_L2s, cmap='YlOrRd', vmin=VMIN, vmax=VMAX)
    fig.colorbar(cf, ax=ax[1,0])
    mean = np.mean(all_L2s)
    std = np.std(all_L2s)
    ax[1,0].set_title("Pixel L2 Loss Heat Map: {:.1f} +/- {:.1f}".format(mean, std), fontsize=tsize)

    # Original image.
    ax[1,1].imshow(I)
    ax[1,1].set_title("An *Example* Image of the Setup", fontsize=tsize)

    # Bells and whistles
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].legend(loc="best", prop={'size':legend_size})
            ax[i,j].set_xlim(xlim)
            ax[i,j].set_ylim(ylim)
            ax[i,j].tick_params(axis='x', labelsize=tick_size)
            ax[i,j].tick_params(axis='y', labelsize=tick_size)

    plt.tight_layout()
    figname = osp.join(OUTPUT_PATH,"check_predictions_scatter_map.png")
    plt.savefig(figname)
    print("Look at this figure:\n{}".format(figname))


def make_plot(ss):
    """Make plot (w/subplots) of losses, etc."""
    nrows, ncols = 2, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(10*ncols,8*nrows), sharey=True)

    all_train = np.array(ss['train'])
    all_test = np.array(ss['test'])
    all_raw_test = np.array(ss['raw_test'])
    all_lrates = np.array(ss['lrates'])
    epochs = ss['epoch']

    # {train, test, raw_test} = shape (K,N) where N is how often we recorded it.
    # For plots, ideally N = E, where E is number of epochs, but usually N > E.
    # For all cross validation folds, we must have N be the same.
    assert all_test.shape == all_raw_test.shape
    K, N = all_train.shape
    print("\nInside make_plot for making some loss curves")
    print("all_train.shape:  {}".format(all_train.shape))
    print("all_test.shape:   {}".format(all_test.shape))
    print("all_lrates.shape: {}".format(all_lrates.shape))
    print("epoch: {}\n".format(epochs))

    # Since N != E in general, try to get epochs to line up.
    xs = np.arange(N) * (epochs[0] / float(N))

    # Plot losses!
    for cv in range(K):
        ax[0,0].plot(xs, all_train[cv,:], label='cv_{}'.format(cv))
        ax[0,1].plot(xs, all_test[cv,:], label='cv_{}'.format(cv))

    # Train = 0, Test = 1.
    mean_0 = np.mean(all_train, axis=0)
    std_0  = np.std(all_train, axis=0)
    mean_1 = np.mean(all_test, axis=0)
    std_1  = np.std(all_test, axis=0)

    train_label = 'Avg CV Losses; minv_{:.4f}'.format( np.min(mean_0) )
    test_label  = 'Avg CV Losses; minv_{:.4f}'.format( np.min(mean_1) )
    ax[1,0].plot(xs, mean_0, lw=2, label=train_label)
    ax[1,1].plot(xs, mean_1, lw=2, label=test_label)
    ax[1,0].fill_between(xs, mean_0-std_0, mean_0+std_0, alpha=error_alpha, facecolor=error_fc)
    ax[1,1].fill_between(xs, mean_1-std_1, mean_1+std_1, alpha=error_alpha, facecolor=error_fc)

    # Titles
    ax[0,0].set_title("CV Train Losses, All Folds (For Debugging Only)", fontsize=tsize)
    ax[0,1].set_title("CV Test Losses, All Folds (For Debugging Only)", fontsize=tsize)
    ax[1,0].set_title("CV Train Losses, Averaged (Scaled, Not Pixels)", fontsize=tsize)
    ax[1,1].set_title("CV Test Losses, Averaged (Scaled, Not Pixels)", fontsize=tsize)

    # Bells and whistles
    for i in range(nrows):
        for j in range(ncols):
            #if LOSS_YLIM is not None:
            #    ax[i,j].set_ylim(LOSS_YLIM)
            ax[i,j].legend(loc="best", ncol=2, prop={'size':legend_size})
            ax[i,j].set_xlabel('Epoch', fontsize=xsize)
            ax[i,j].set_ylabel('Average L2 Loss (Scaled)', fontsize=ysize)
            ax[i,j].set_ylabel('Average L2 Loss (Scaled)', fontsize=ysize)
            ax[i,j].tick_params(axis='x', labelsize=tick_size)
            ax[i,j].tick_params(axis='y', labelsize=tick_size)

    plt.tight_layout()
    figname = osp.join(OUTPUT_PATH,"check_stats.png")
    plt.savefig(figname)
    print("Look at this figure:\n{}".format(figname))


def plot_data_source(ss):
    """Make plot w.r.t. data source, for combo data.

    Please modify DSOURCES at the top. Sorry for sloppy research code.
    Actually we won't have a plot unfortunately, we just print some stuff ...
    """
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(10*ncols,8*nrows), sharey=True)

    # Global information
    all_train = np.array(ss['train'])
    all_test = np.array(ss['test'])
    all_raw_test = np.array(ss['raw_test'])
    all_lrates = np.array(ss['lrates'])
    epochs = ss['epoch']

    # Deal with individual points for data sources
    all_d_sources = ss['data_sources']
    all_L2s       = np.array(ss['all_L2s'])

    # {train, test, raw_test} = shape (K,N) where N is how often we recorded it.
    # For plots, ideally N = E, where E is number of epochs, but usually N > E.
    # For all cross validation folds, we must have N be the same.
    assert all_test.shape == all_raw_test.shape
    K, N = all_train.shape
    print("\nInside plot_data_source for analysis of performance wrt individual data sources")
    print("all_train.shape:  {}".format(all_train.shape))
    print("all_test.shape:   {}".format(all_test.shape))
    print("all_L2s.shape:    {}".format(all_L2s.shape))
    print("len(all_d_sources):    {}".format(len(all_d_sources)))
    for idx,val in enumerate(all_d_sources):
        print("  len(all_d_sources[{}]): {}".format(idx,len(all_d_sources[idx])))
    print("epoch: {}\n".format(epochs))

    # Since N != E in general, try to get epochs to line up.
    xs = np.arange(N) * (epochs[0] / float(N))

    # Need to split data among the two data sources.
    flat_data_list = [item for sublist in all_d_sources for item in sublist]
    ds0 = []
    ds1 = []
    for idx,(ds,L2) in enumerate(zip(flat_data_list,all_L2s)):
        if ds == DSOURCES[0]:
            ds0.append(L2)
        elif ds == DSOURCES[1]:
            ds1.append(L2)
        else:
            raise ValueError(ds)
    print("For {}:".format(DSOURCES[0]))
    print("   avg: {:.1f}\pm {:.1f}".format( np.mean(ds0), np.std(ds0) ))
    print("For {}:".format(DSOURCES[1]))
    print("   avg: {:.1f}\pm {:.1f}".format( np.mean(ds1), np.std(ds1) ))



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
            c_suffix = 'r_cv_{}_grasp_{}_rgb_L2_{:.0f}.png'.format(cv_index, idx, L2)
            d_suffix = 'r_cv_{}_grasp_{}_depth_L2_{:.0f}.png'.format(cv_index, idx, L2)
            if 'combo' in DATA_NAME:
                data_s = data_sources[idx]
                c_suffix = 'r_cv_{}_grasp_{}_rgb_L2_{:.0f}_{}.png'.format(cv_index, idx, L2, data_s)
                d_suffix = 'r_cv_{}_grasp_{}_depth_L2_{:.0f}_{}.png'.format(cv_index, idx, L2, data_s)
            c_path = osp.join(OUTPUT_PATH, c_suffix)
            d_path = osp.join(OUTPUT_PATH, d_suffix)

            # For plotting later. Note, `targ` is the ground truth.
            ss['all_targs'].append(targ)
            ss['all_preds'].append(pred)
            ss['all_x'].append(targ[0])
            ss['all_y'].append(targ[1])
            ss['all_L2s'].append(L2)
            ss['all_names'].append(d_path)
            targ  = (int(targ[0]), int(targ[1]))
            preds = (int(pred[0]), int(pred[1]))

            # Overlay the pose to the image (red circle, black border).
            cv2.circle(cimg, center=targ, radius=INNER, color=(0,0,255), thickness=-1)
            cv2.circle(dimg, center=targ, radius=INNER, color=(0,0,255), thickness=-1)
            cv2.circle(cimg, center=targ, radius=OUTER, color=(0,0,0), thickness=1)
            cv2.circle(dimg, center=targ, radius=OUTER, color=(0,0,0), thickness=1)

            # The PREDICTION, though, will be a large blue circle (yellow border?).
            cv2.circle(cimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
            cv2.circle(dimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
            cv2.circle(cimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)
            cv2.circle(dimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)

            cv2.imwrite(c_path, cimg)
            cv2.imwrite(d_path, dimg)

        # Add some more stuff about this cv set, e.g., loss curves.
        ss['train'].append(data_other['train'])
        ss['test'].append(data_other['test'])
        ss['raw_test'].append(data_other['raw_test'])
        ss['epoch'].append(data_other['epoch'])
        ss['lrates'].append(data_other['lrates'])

        print("=====================================================================")

    print("\nDone with creating overlays, look at:\n{}\nfor all the predictions".format(
            OUTPUT_PATH))
    if 'combo' in DATA_NAME:
        plot_data_source(ss)
    make_scatter(ss)
    make_plot(ss)
    scatter_heat_final(ss)
