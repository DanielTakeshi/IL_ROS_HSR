"""Plot the training from Ron's data.

Specifically, this will assume dictionaries accessible in a directory like this:

(py2-bedmake) seita@triton2:/nfs/diskstation/seita/bed-make/grasp_output$ ls -lh stats/
total 80K
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_0.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_1.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_2.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_3.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_4.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_5.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_6.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_7.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_8.p
grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001_cv_9.p

and these have data (from `fast_grasp_detect/core/train_network.py`) that I can use for figures.

With Ron's data, I had time to do 10-fold cross validation so this will contain averages, unlike the
previous one I did with Michael's NYTimes data and my Cal data.

Adjust directories in the MAIN METHOD at the bottom of this script!
"""
import os, pickle, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np

# Matplotlib
plt.style.use('seaborn-darkgrid')
error_region_alpha = 0.25
title_size = 22
tick_size = 17
legend_size = 17
xsize = 18
ysize = 18
lw = 3
ms = 8


def get_info(head, logs):
    stats = []
    for pfile_name in logs:
        # Should have keys: ['test', 'epoch', 'train', 'raw_test', 'name'].
        data = pickle.load( open(os.path.join(head,pfile_name)) )
        epochs = data['epoch']
        K = len(data['test'])
        assert K == len(data['raw_test'])
        factor = float(epochs) / K
        print("epochs: {}, len(data['test']): {}".format(epochs, K))
        stats.append( np.array(data['test']) )

    # I actually kept multiple values per epoch.
    xcoord = np.arange(K) * factor

    return np.array(stats), xcoord


def plot(head_b, head_0, head_1, figname):
    """For now we used three different training data.

    Both heights 0 and 1, or just height 0, or just height 1.  We might further
    split it based on arm height, but we don't need to worry about that for now.

    BTW I realize I didn't run with as many epochs for the larger dataset but
    that's a minor thing, I don't think results would improve that much.
    """
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols, 8*nrows),
            squeeze=False, sharex=False, sharey=True)

    # Collect information about the three different training datasets.
    logs_0 = sorted([x for x in os.listdir(head_0) if x[-2:]=='.p'])
    logs_1 = sorted([x for x in os.listdir(head_1) if x[-2:]=='.p'])
    logs_b = sorted([x for x in os.listdir(head_b) if x[-2:]=='.p'])

    stats_0, xcoord_0 = get_info(head_0, logs_0)
    print("Loaded {}. Shape: {}".format(head_0, stats_0.shape))
    label_0 = 'camera-class0'

    stats_1, xcoord_1 = get_info(head_1, logs_1)
    print("Loaded {}. Shape: {}".format(head_1, stats_1.shape))
    label_1 = 'camera-class1'

    stats_b, xcoord_b = get_info(head_b, logs_b)
    print("Loaded {}. Shape: {}".format(head_b, stats_b.shape))
    label_b = 'both-c-angles'

    color='red'
    mean = np.mean(stats_0, axis=0)
    std = np.std(stats_0, axis=0)
    label_0 += '-min-{:.4f}'.format(np.min(mean))
    axes[0,0].plot(xcoord_0, mean, lw=lw, color=color, label=label_0)
    axes[0,0].fill_between(xcoord_0, mean-std, mean+std, alpha=error_region_alpha, facecolor=color)

    color='yellow'
    mean = np.mean(stats_1, axis=0)
    std = np.std(stats_1, axis=0)
    label_1 += '-min-{:.4f}'.format(np.min(mean))
    axes[0,0].plot(xcoord_1, mean, lw=lw, color=color, label=label_1)
    axes[0,0].fill_between(xcoord_1, mean-std, mean+std, alpha=error_region_alpha, facecolor=color)

    color='blue'
    mean = np.mean(stats_b, axis=0)
    std = np.std(stats_b, axis=0)
    label_b += '-min-{:.4f}'.format(np.min(mean))
    axes[0,1].plot(xcoord_b, mean, lw=lw, color=color, label=label_b)
    axes[0,1].fill_between(xcoord_b, mean-std, mean+std, alpha=error_region_alpha, facecolor=color)

    axes[0,0].set_title('Test Raw L2 Losses, Split Data', size=title_size)
    axes[0,1].set_title('Test Raw L2 Losses, Combo Data', size=title_size)

    # Bells and whistles.
    for rr in range(nrows):
        for cc in range(ncols):
            axes[rr,cc].tick_params(axis='x', labelsize=tick_size)
            axes[rr,cc].tick_params(axis='y', labelsize=tick_size)
            axes[rr,cc].legend(loc="best", prop={'size':legend_size})
            axes[rr,cc].set_xlabel('Training Epochs', size=xsize)
            axes[rr,cc].set_ylabel('Test Loss (Raw, L2 Pixels)', size=ysize)
            #axes[rr,cc].set_yscale('log')
            axes[rr,cc].set_ylim([0.0,0.04])
    fig.tight_layout()
    fig.savefig(figname)
    print("\nJust saved: {}\n".format(figname))


if __name__ == "__main__":
    head_b  = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/ron_v02_b_fix26'
    head_0  = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/ron_v02_c0_fix26'
    head_1  = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/ron_v02_c1_fix26'
    figname = '/nfs/diskstation/seita/bed-make/figures/train_exploration_ron.png'
    plot(head_b, head_0, head_1, figname)
