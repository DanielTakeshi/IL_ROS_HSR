"""Use this script for inspecting results after we run.

E.g. after running:

    ./main/grasp.sh | tee logs/grasp.log

for either cross validation or not, I should be able to automate things. These should create output
that lie under a directory, like: `grasp/ABC/[results...]`. And don't forget figures too.
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
# ADJUST. HH is the directory named like: 'grasp_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/'
DATA_NAME = 'rollouts_white_v01'
HH = 'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_cv_True'
ROLLOUT_HEAD = osp.join(HEAD, DATA_NAME)

# Sanity checks.
rgb_baseline = utils.rgb_baseline_check(HH)
net_type = utils.net_check(HH)

# Make directory. In separate `figures/`, we put the same directory name for results.
RESULTS_PATH = osp.join(HEAD, net_type, DATA_NAME, HH)
OUTPUT_PATH  = osp.join(HEAD, 'figures', DATA_NAME, HH)
if not osp.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Sizes for overlaying predictions/targets. Maybe cross hair is better?
INNER, OUTER = 3, 4

# For the plot(s). There are a few plot-specific parameters, though.
tsize = 20
xsize = 18
ysize = 18
tick_size = 18
legend_size = 18
alpha = 0.5
error_alpha = 0.3
error_fc = 'gold'
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def make_scatter(ss):
    """Make giant scatter plot image."""
    nrows, ncols = 2, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(8*nrows,6*ncols))
    xlim = [0, 640]
    ylim = [480, 0]
    VMIN = 0
    VMAX = 200

    # Put all stuff locally from the dictionary input (turning into np.arrays as needed).
    all_targs = np.array(ss['all_targs'])
    all_preds = np.array(ss['all_preds'])
    all_L2s = np.array(ss['all_L2s'])
    all_top = ss['all_top']
    all_bot = ss['all_bot']
    all_x = ss['all_x']
    all_y = ss['all_y']
    all_names = ss['all_names']

    num_pts = len(ss['all_L2s'])
    print("\nall_targs.shape: {}".format(all_targs.shape))
    print("all_preds.shape: {}".format(all_preds.shape))
    print("all_L2s.shape:   {}".format(all_L2s.shape))
    print("all_L2s (pixels): {:.1f} +/- {:.1f}".format(np.mean(all_L2s), np.std(all_L2s)))
    print("TOP L2s (pixels): {:.1f} +/- {:.1f};  max {:.1f}, quantity {}".format(
            np.mean(all_top), np.std(all_top), np.max(all_top), len(all_top)))
    print("BOT L2s (pixels): {:.1f} +/- {:.1f};  max {:.1f}, quantity {}".format(
            np.mean(all_bot), np.std(all_bot), np.max(all_bot), len(all_bot)))
    
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
    I = cv2.imread("scripts/imgs/image_example.png")
    
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
    ax[1,0].set_title("L2 Losses (Heat Map) in Pixels: {:.1f} +/- {:.1f}".format(mean, std), fontsize=tsize)
    
    # Original image.
    ax[1,1].imshow(I)
    ax[1,1].set_title("*Example* Image of the Setup", fontsize=tsize)
    
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
    print("Look at this figure: {}".format(figname))
    

def make_plot(ss):
    """Make plot (w/subplots) of losses, etc."""
    nrows, ncols = 2, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(8*nrows,6*ncols),
            sharey=True)

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
    print("\nall_train.shape:  {}".format(all_train.shape))
    print("all_test.shape:   {}".format(all_test.shape))
    print("all_lrates.shape: {}".format(all_lrates.shape))
    print("epoch: {}\n".format(epochs))

    # Since N != E in general, try to get epochs to line up.
    xs = np.arange(N) * (epochs[0] / float(N))

    # Plot losses!
    for cv in range(K):
        ax[0,0].plot(xs, all_train[cv,:], label='cv_{}'.format(cv))
        ax[0,1].plot(xs, all_test[cv,:], label='cv_{}'.format(cv))

    mean_0 = np.mean(all_train, axis=0)
    std_0  = np.std(all_train, axis=0)
    mean_1 = np.mean(all_test, axis=0)
    std_1  = np.std(all_test, axis=0)

    ax[1,0].plot(xs, mean_0, lw=2, label='train_losses')
    ax[1,1].plot(xs, mean_1, lw=2, label='test_losses')
    ax[1,0].fill_between(xs, mean_0-std_0, mean_0+std_0, alpha=error_alpha, facecolor=error_fc)
    ax[1,1].fill_between(xs, mean_1-std_1, mean_1+std_1, alpha=error_alpha, facecolor=error_fc)

    # Titles
    ax[0,0].set_title("CV Train Losses", fontsize=tsize)
    ax[0,1].set_title("CV Test Losses", fontsize=tsize)
    ax[1,0].set_title("CV Train Losses", fontsize=tsize)
    ax[1,1].set_title("CV Test Losses", fontsize=tsize)

    # Bells and whistles
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].legend(loc="best", ncol=2, prop={'size':legend_size})
            ax[i,j].set_xlabel('Epoch', fontsize=xsize)
            ax[i,j].set_ylabel('Average L2 Loss', fontsize=ysize)
            ax[i,j].tick_params(axis='x', labelsize=tick_size)
            ax[i,j].tick_params(axis='y', labelsize=tick_size)
    
    plt.tight_layout()
    figname = osp.join(OUTPUT_PATH,"check_stats.png")
    plt.savefig(figname)
    print("Look at this figure: {}".format(figname))
 

if __name__ == "__main__":
    pfiles = sorted([x for x in os.listdir(RESULTS_PATH) if '_raw_imgs.p' in x])
    ss = defaultdict(list) # for plotting later
    
    for pf in pfiles:
        # Now on one of the cross-validation splits (or a 'normal' training run).
        other_pf = pf.replace('_raw_imgs.p','.p')
        print("\n\n\n ----- Now on: {}\n ----- with:   {}".format(pf, other_pf))
        data_imgs  = pickle.load( open(osp.join(RESULTS_PATH,pf),'rb') )
        data_other = pickle.load( open(osp.join(RESULTS_PATH,other_pf),'rb') )
    
        # Load predictions, targets, and images.
        y_pred = data_other['preds']
        y_targ = data_other['targs']
        c_imgs = data_imgs['c_imgs_list']
        d_imgs = data_imgs['d_imgs_list']
        assert len(y_pred) == len(y_targ) == len(c_imgs) == len(d_imgs)
    
        # Later, figure out what to do if not using cross validation ...
        if 'cv_indices' in data_other:
            cv_ids = data_other['cv_indices']
            print("Now dealing with CV (rollout) indices: {}".format(cv_ids))
    
        # Index into y_pred, y_targ, c_imgs, d_imgs, _within_ this CV training run.
        idx = 0
    
        for rnum in cv_ids:
            print("\n=====================================================================")
            path = osp.join(ROLLOUT_HEAD, 'rollout_{}/rollout.p'.format(rnum))
            if not osp.exists(path):
                print("Error: {} does not exist".format(path))
                sys.exit()
            data = pickle.load(open(path,'rb'))
            print("loaded: {}".format(path))
            print("rollout {}, len(data): {}".format(rnum, len(data)))
    
            # Unfortunately we'll assume that we have two grasps. For now we know this is the case, but
            # we can also load the rollout (as we do) and further inspect within that just to confirm.
            assert len(data) == 4
            assert data[0]['type'] == 'grasp'
            assert data[1]['type'] == 'success'
            assert data[2]['type'] == 'grasp'
            assert data[3]['type'] == 'success'
    
            for g_in_r in range(2):
                # Get these from our training run, stored from best iteration on validation set.
                pred = y_pred[idx]
                targ = y_targ[idx]
                cimg = c_imgs[idx].copy()
                dimg = d_imgs[idx].copy()
                idx += 1
                l2 = np.sqrt( (pred[0]-targ[0])**2 + (pred[1]-targ[1])**2)
                c_suffix = 'r_{}_grasp_{}_rgb_L2_{:.0f}.png'.format(rnum,g_in_r,l2)
                d_suffix = 'r_{}_grasp_{}_depth_L2_{:.0f}.png'.format(rnum,g_in_r,l2)
                c_path = osp.join(OUTPUT_PATH, c_suffix)
                d_path = osp.join(OUTPUT_PATH, d_suffix)
    
                # For visualization and inspection later; good idea, Ron, Ajay, Honda :-).
                side = data[g_in_r*2]['side']
                if side == 'BOTTOM':
                    ss['all_bot'].append(l2)
                elif side == 'TOP':
                    ss['all_top'].append(l2)
                else:
                    raise ValueError(side)
                ss['all_targs'].append(targ)
                ss['all_preds'].append(pred)
                ss['all_x'].append(targ[0])
                ss['all_y'].append(targ[1])
                ss['all_L2s'].append(l2)
                ss['all_names'].append(d_path)
    
                # Alternatively could get from rollout paths. Good to double check. Unfortunately again
                # this assumes I did grasp then success then grasp then success ... yeah.
                pose  = data[g_in_r*2]['pose']
                pose  = (int(pose[0]), int(pose[1]))
                targ  = (int(targ[0]), int(targ[1]))
                preds = (int(pred[0]), int(pred[1]))
                print("pose: {}, targ: {} (should match), w/preds {}".format(pose, targ, preds))
    
                # Overlay the pose to the image (red circle, black border).
                cv2.circle(cimg, center=pose, radius=INNER, color=(0,0,255), thickness=-1)
                cv2.circle(dimg, center=pose, radius=INNER, color=(0,0,255), thickness=-1)
                cv2.circle(cimg, center=pose, radius=OUTER, color=(0,0,0), thickness=1)
                cv2.circle(dimg, center=pose, radius=OUTER, color=(0,0,0), thickness=1)
        
                # The PREDICTION, though, will be a large blue circle (yellow border?).
                cv2.circle(cimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
                cv2.circle(dimg, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
                cv2.circle(cimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)
                cv2.circle(dimg, center=preds, radius=OUTER, color=(0,255,0), thickness=1)
        
                cv2.imwrite(c_path, cimg)
                cv2.imwrite(d_path, dimg)
            
        # Add some more stuff about this cv set, e.g. loss curves.
        ss['train'].append(data_other['train'])
        ss['test'].append(data_other['test'])
        ss['raw_test'].append(data_other['raw_test'])
        ss['epoch'].append(data_other['epoch'])
        ss['lrates'].append(data_other['lrates'])

        print("=====================================================================")

    print("\nDone with creating overlays, look at {}".format(OUTPUT_PATH))
    make_scatter(ss)
    make_plot(ss)
