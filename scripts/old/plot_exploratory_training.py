"""Plot the training runs I do.

Specifically, this will assume dictionaries accessible in a directory like this:

(py2-bedmake) seita@triton2:/nfs/diskstation/seita/bed-make$ ls -lh grasp_output_for_plots/nytimes/
total 96K
-rw-rw-r-- 1 nobody nogroup 9.8K Jul 11 19:31 grasp_net_depth_True_optim_ADAM_fixed_False_lrate_0.0001.p
-rw-rw-r-- 1 nobody nogroup 9.8K Jul 11 19:39 grasp_net_depth_True_optim_ADAM_fixed_False_lrate_1e-05.p
-rw-rw-r-- 1 nobody nogroup 9.8K Jul 11 19:47 grasp_net_depth_True_optim_ADAM_fixed_True_lrate_0.0001.p
-rw-rw-r-- 1 nobody nogroup 9.9K Jul 11 19:54 grasp_net_depth_True_optim_ADAM_fixed_True_lrate_1e-05.p
-rw-rw-r-- 1 nobody nogroup 9.8K Jul 11 20:09 grasp_net_depth_True_optim_SGD_fixed_False_lrate_0.01.p
-rw-rw-r-- 1 nobody nogroup 9.7K Jul 11 20:03 grasp_net_depth_True_optim_SGD_fixed_False_lrate_0.1.p
(and so on)

and these have data (from `fast_grasp_detect/core/train_network.py`) that I can use for figures.

For now, I have tests involving 32 setups, 16 with my Cal data and 16 with Michael's NYTimes data. I
will make two figures here, each of which have two subplots (so, four plots total). This will pack
lots of information together.

This is mostly for exploring the effect of different hyperparameters.

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


def parse_name(name):
    """Given a file name, we parse the file to return relevant stuff for legends, etc."""
    name_split = name.split('_')
    assert name_split[0] == 'grasp' and name_split[1] == 'net', "name_split: {}".format(name_split)
    assert name_split[2] == 'depth' and name_split[4] == 'optim', "name_split: {}".format(name_split)
    assert name_split[6] == 'fixed' and name_split[8] == 'lrate', "name_split: {}".format(name_split)
    info = {'depth': name_split[3],
            'optim': (name_split[5]).lower(),
            'fixed': name_split[7],
            'lrate': (name_split[9]).rstrip('.p')}
    # Actually might as well do the legend here. Start with depth vs rgb, etc.
    if info['depth'] == 'True':
        legend = 'depth-'
    else:
        assert info['depth'] == 'False'
        legend = 'rgb-'
    if info['fixed'] == 'True':
        legend += 'fix26-'
    else:
        assert info['fixed'] == 'False'
        legend += 'train26-'
    legend += 'opt-{}-lr-{}'.format(info['optim'], info['lrate'])
    return info, legend


def plot(head, figname):
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols, 8*nrows),
            squeeze=False, sharex=True, sharey=True)

    # Load and parse information.
    logs = sorted([x for x in os.listdir(head) if x[-2:]=='.p'])
    print("Loading these logs:")

    for pickle_file_name in logs:
        _, legend_l = parse_name(pickle_file_name)
        print("{}  (legend: {})".format(pickle_file_name, legend_l))

        # Should have keys: ['test', 'epoch', 'train', 'raw_test', 'name'].
        data = pickle.load( open(os.path.join(head,pickle_file_name)) )
        epochs = data['epoch']
        K = len(data['test'])
        assert K == len(data['raw_test'])
        factor = float(epochs) / K
        print("epochs: {}, len(data['test']): {}".format(epochs, K))
        xcoord = np.arange(K) * factor # I actually kept multiple values per epoch.

        # Add min value to legend label
        legend_l += '-min-{:.4f}'.format(np.min(data['test']))

        # For plotting test loss, I actually kept multiple values per epoch. Just take the min.
        cc = 0
        if 'depth-' in legend_l:
            cc = 1
        #axes[0,cc].plot(xcoord, data['raw_test'], lw=lw, label=legend_l) # raw pixels
        axes[0,cc].plot(xcoord, data['test'], lw=lw, label=legend_l) # Or scaled

    axes[0,0].set_title('Test Losses, RGB Images Only', size=title_size)
    axes[0,1].set_title('Test Losses, Depth Images Only', size=title_size)

    # Bells and whistles.
    for rr in range(nrows):
        for cc in range(ncols):
            axes[rr,cc].tick_params(axis='x', labelsize=tick_size)
            axes[rr,cc].tick_params(axis='y', labelsize=tick_size)
            axes[rr,cc].legend(loc="best", prop={'size':legend_size})
            axes[rr,cc].set_xlabel('Training Epochs', size=xsize)
            axes[rr,cc].set_ylabel('Test Loss (Raw, L2 Pixels)', size=ysize)
            axes[rr,cc].set_yscale('log')
            axes[rr,cc].set_ylim([0.003,0.1])
    fig.tight_layout()
    fig.savefig(figname)
    print("\nJust saved: {}\n".format(figname))


if __name__ == "__main__":
    head1 = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/nytimes'
    head2 = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/danielcal'
    figname1 = '/nfs/diskstation/seita/bed-make/figures/train_exploration_nytimes.png'
    figname2 = '/nfs/diskstation/seita/bed-make/figures/train_exploration_danielcal.png'
    plot(head1, figname1)
    plot(head2, figname2)
