"""Use this script for inspecting results after doing bed-making deployment.

See the bed-making deployment code for how we saved things.
There are lots of things we can do for inspection.
"""
import argparse, cv2, os, pickle, sys, matplotlib, utils
from os.path import join
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
np.set_printoptions(suppress=True, linewidth=200, precision=4)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from collections import defaultdict

# ------------------------------------------------------------------------------
# ADJUST. HH is directory like: 'grasp_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/results/'
RESULTS = join(HEAD, 'deploy_network')
FIGURES = join(HEAD, 'figures')

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


def analyze_time(stats):
    """For analyzing time. These should be in seconds unless otherwise stated.
    """
    print("\nTimes for moving to other side:")
    move_t = np.array( stats['move_times'] )
    print(move_t)
    print("\nTimes for executing grasps:")
    grasp_t = np.array( stats['grasp_times'] )
    print(grasp_t)

    grasp_net_t = []
    success_net_t = []
    result = stats['result_0'] # only one rollout here

    for i,info in enumerate(result):
        if result[i]['type'] == 'grasp':
            grasp_net_t.append( result[i]['g_net_time'] )
        else:
            assert result[i]['type'] == 'success'
            success_net_t.append( result[i]['s_net_time'] )

    grasp_net_t = np.array(grasp_net_t)
    success_net_t = np.array(success_net_t)
    print("\ngrasp_net_t:\n{}".format(grasp_net_t))
    print("{:.3f} +/- {:.3f}".format(np.mean(grasp_net_t), np.std(grasp_net_t)))
    print("\nsuccess_net_t:\n{}".format(success_net_t))
    print("{:.3f} +/- {:.3f}".format(np.mean(success_net_t), np.std(success_net_t)))


def analyze_preds(stats):
    """For analyzing predictions.
    """
    result = stats['result_0']
    # do something with the images, overlay them, etc.?


if __name__ == "__main__":
    p_files = sorted([join(RESULTS,x) for x in os.listdir(RESULTS) if 'results_rollout' in x])

    # stuff info here for plotting, etc.
    stats = defaultdict(list)

    for p_idx, p_file in enumerate(p_files):
        with open(p_file, 'r') as f:
            data = pickle.load(f)
        print("\n==============================================================")
        print("loaded file #{} at {}".format(p_idx, p_file))

        # All items except last one should reflect some grasp or success nets.
        key = 'result_{}'.format(p_idx)
        stats[key] = data[:-1]
       
        # We know the final dict has some useful stuff in it
        final_dict = data[-1]
        assert 'move_times' in final_dict and 'grasp_times' in final_dict \
                and 'final_c_img' in final_dict
        stats['move_times'].append( final_dict['move_times'] )
        stats['grasp_times'].append( final_dict['grasp_times'] )

    analyze_time(stats)
    analyze_preds(stats)
