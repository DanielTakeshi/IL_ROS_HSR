"""Use this script for inspecting results after we run.

The success net, that is. I don't think we'll have figures, but tables are helpful.
We can just report correctness as a function of training epoch, in a table.
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
# ADJUST. HH is directory like: 'success_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/'
DATA_NAME = 'cache_combo_v01_success'
HH = 'success_4_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_3000_cv_True'

# Sanity checks.
net_type = utils.net_check(HH)
VIEW_TYPE = 'standard'
assert VIEW_TYPE in ['standard', 'close']
assert 'cache' in DATA_NAME

# Make directory. In separate `figures/`, we put the same directory name for results.
RESULTS_PATH = osp.join(HEAD, net_type, DATA_NAME, HH)
OUTPUT_PATH  = osp.join(HEAD, 'figures', DATA_NAME, HH)
if not osp.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# For the plot(s). There are a few plot-specific parameters, though.
tsize = 30
xsize = 25
ysize = 25
tick_size = 25
legend_size = 25
alpha = 0.5
error_alpha = 0.3
error_fc = 'blue'
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def plot(ss):
    """Make plot (w/subplots) of performance, etc.

    For now this is just plotting 'correctness', actually as a fraction.
    I don't deal with the training data, the network is basically perfect.
    """
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(10*ncols,8*nrows), sharey=True, squeeze=False)

    all_correctness = np.array(ss['all_correctness'])
    epochs = ss['epoch']
    num_imgs = ss['total_imgs']
    print("Now plotting, with all_correctness.shape: {}".format(all_correctness.shape))
    print("num_imgs: {}".format(num_imgs)) # different folds may have number differ by 1

    # all_correctness = shape (K,N) where N is how often we recorded it.
    # For plots, ideally N = E, where E is number of epochs, but usually N > E.
    # For all cross validation folds, we must have N be the same.
    K, N = all_correctness.shape

    # Easier for us to do accuracy here so scale by this.
    num_imgs = np.reshape(num_imgs, (K,1))

    # Each 'row' of the matrix consists of percentage correct at each time we recorded it.
    # e.g., [0.64, 0.77, 0.84, 0.91, 0.94, ...], and hopefully to 1.0 :-).
    correctness_scaled = all_correctness / num_imgs

    # Since N != E in general, try to get epochs to line up.
    xs = np.arange(N) * (epochs[0] / float(N))

    # Finally, plot. First do individual curves:
    for cv in range(K):
        ax[0,0].plot(xs, correctness_scaled[cv,:], lw=2, label='cv_{}'.format(cv))

    # Then do the average with error regions, my favorite.
    mean = np.mean(correctness_scaled, axis=0)
    std = np.std(correctness_scaled, axis=0)
    label = 'Avg CV Correct; maxv_{:.4f}'.format( np.max(mean) )
    ax[0,1].plot(xs, mean, lw=2, label=label)
    ax[0,1].fill_between(xs, mean-std, mean+std, alpha=error_alpha, facecolor=error_fc)

    # Titles
    ax[0,0].set_title("CV Test Losses, Correctness (All)", fontsize=tsize)
    ax[0,1].set_title("CV Test Losses, Correctness (Mean)", fontsize=tsize)

    # Bells and whistles
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].set_ylim([0.0, 1.0])
            ax[i,j].legend(loc="best", ncol=2, prop={'size':legend_size})
            ax[i,j].set_xlabel('Training Epochs Over Augmented Data', fontsize=xsize)
            ax[i,j].set_ylabel('Fraction Correct', fontsize=ysize)
            ax[i,j].tick_params(axis='x', labelsize=tick_size)
            ax[i,j].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    figname = osp.join(OUTPUT_PATH,"check_correctness.png")
    plt.savefig(figname)
    print("Look at this figure:\n{}".format(figname))

    # NOTE: use this in a paper if we want to report correctness over a few epochs
    # Of course, a plot might be good but it's a lower priority for the success net.
    # Advantage of this approach is that we average over same, fixed epochs, so it's
    # not 'cheating' by taking the best epoch independently for each CV fold.

    print("\nmaybe use to refer to quick epoch : accuracy numbers?\n")
    result = [(i1,i2) for (i1,i2) in zip(xs, np.sum(all_correctness, axis=0))]
    for item in result:
        print(item)


if __name__ == "__main__":
    print("RESULTS_PATH: {}".format(RESULTS_PATH))
    pfiles = sorted([x for x in os.listdir(RESULTS_PATH) if '_raw_imgs.p' in x])
    ss = defaultdict(list) # for plotting later
    num_incorrect = 0

    for cv_index,pf in enumerate(pfiles):
        # Now on one of the cross-validation splits (or a 'normal' training run).
        other_pf = pf.replace('_raw_imgs.p','.p')
        print("\n\n ----- Now on: {}\n ----- with:   {}".format(pf, other_pf))
        data_imgs  = pickle.load( open(osp.join(RESULTS_PATH,pf),'rb') )
        data_other = pickle.load( open(osp.join(RESULTS_PATH,other_pf),'rb') )

        # Load various stuff we stored from training.
        correct = data_other['success_test_correct'] # list of total correct
        acc = data_other['best_snet_acc'] # best fraction (i.e., correct/total)
        preds = data_other['best_snet_preds'] # best preds (each 2D) for best_acc
        correctness = data_other['best_snet_correctness'] # individual image performance

        c_imgs = data_imgs['c_imgs_list']
        d_imgs = data_imgs['d_imgs_list']
        K = len(c_imgs)
        assert len(correctness) == K

        # Later, figure out what to do if not using cross validation ...
        if 'cv_indices' in data_other:
            cv_fname = data_other['cv_indices']
            print("Now processing CV file name: {}, idx {}, with {} images".format(
                    cv_fname, cv_index, K))

        # `idx` = index into stuff _within_ this CV test set. Save images w/label.
        for idx in range(K):
            if idx % 10 == 0:
                print("  ... processing image {} in this test set".format(idx))
            is_it_correct = correctness[idx]
            prediction = preds[idx]
            cimg = c_imgs[idx].copy()
            dimg = d_imgs[idx].copy()
            result = 'good'
            if not is_it_correct:
                result = 'INCORRECT'
                num_incorrect += 1
            c_suffix = 'r_cv_{}_s-net_{}_rgb_{}.png'.format(cv_index, idx, result)
            d_suffix = 'r_cv_{}_s-net_{}_depth_{}.png'.format(cv_index, idx, result)
            c_path = osp.join(OUTPUT_PATH, c_suffix)
            d_path = osp.join(OUTPUT_PATH, d_suffix)
            cv2.imwrite(c_path, cimg)
            cv2.imwrite(d_path, dimg)

        # For plotting, we might want some of this stuff.
        ss['all_correctness'].append(correct)
        ss['all_predictions'].append(preds)
        ss['train'].append(data_other['train'])
        ss['test'].append(data_other['test'])
        ss['epoch'].append(data_other['epoch'])
        ss['lrates'].append(data_other['lrates'])
        ss['total_imgs'].append(K)
        print("=====================================================================")

    print("\nDone with iterating through images. See:\n{}".format(OUTPUT_PATH))
    print("num_incorrect: {}".format(num_incorrect))
    print("This is the best per CV ... if we fix one epoch and evaluate the CVs")
    print("all at that epoch, we'll likely get a few more incorrects.\n")
    print("For a paper, consider just recording the #correct after every few epochs in a table")
    print("No need for overly fancy plots.\n")
    plot(ss)
