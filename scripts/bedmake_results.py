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
# ADJUST.
# ------------------------------------------------------------------------------
RESULTS = '/nfs/diskstation/seita/bed-make/results/'

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


if __name__ == "__main__":
    print("Searching in path: {}".format(RESULTS))
    PATHS = sorted(
        [join(RESULTS,x) for x in os.listdir(RESULTS) 
         if 'deploy_' in x and 'old' not in x]
    )
    print("Looking at these paths:")
    for item in PATHS:
        print('  '+item)
    print("")

    # Move times, should be pretty slow. Measured for all.
    move_t = []

    # Robot grasp execution. Also slow, can be highly variable, measured for all.
    grasp_t = []

    # Now times for grasping and success net. These are quick. Only for networks.
    # Actually for now, I just combine them.
    grasp_net_t = []
    success_net_t = []
 
    for pth in PATHS:
        # The `pth` is `deploy_network_white`, `deploy_human`, etc.
        print("\n=========================================================================")
        print("Now on: {}".format(pth))
        rollouts = sorted([join(pth,x) for x in os.listdir(pth) if 'results_rollout' in x])
        if len(rollouts) == 0:
            print("len(rollouts) == 0, thus skipping this ...")
            continue

        stats = defaultdict(list)
        num_grasps = []

        for r_idx,rollout_pfile in enumerate(rollouts):
            with open(rollout_pfile, 'r') as f:
                data = pickle.load(f)

            # All items except last one should reflect some grasp or success nets.
            # We know the final dict has some useful stuff in it for timing.

            final_dict = data[-1]
            assert 'move_times' in final_dict and 'grasp_times' in final_dict \
                    and 'image_start' in final_dict and 'image_final' in final_dict
            stats['move_times'].append( final_dict['move_times'] )
            stats['grasp_times'].append( final_dict['grasp_times'] )

            # Debugging
            gtimes = final_dict['grasp_times']
            mtimes = final_dict['move_times']
            print("{}".format(rollout_pfile))
            print("    g-times:  {}".format(np.array(gtimes)))
            print("    m-times:  {}".format(np.array(mtimes)))
            num_grasps.append( len(gtimes) )

            # Add statistics about grasping times.
            if '_network_' in pth:
                g_net = []
                s_net = []

                # Go up to the last one, which has the timing data we just collected.
                # Here we are only interested in the network forward pass times.
                for i,datum in enumerate(data[:-1]):
                    if datum['type'] == 'grasp':
                        grasp_net_t.append( datum['g_net_time'] )
                        g_net.append( datum['g_net_time'] )
                    else:
                        assert datum['type'] == 'success'
                        success_net_t.append( datum['s_net_time'] )
                        s_net.append( datum['s_net_time'] )

                print("    g-net-times:  {}".format(np.array(g_net)))
                print("    s-net-times:  {}".format(np.array(s_net)))

        print("num grasps: {:.1f} \pm {:.1f}".format(np.mean(num_grasps),np.std(num_grasps)))

        # Add these to the global lists.
        for stats_l in stats['move_times']:
            for tt in stats_l:
                move_t.append(tt)
        for stats_l in stats['grasp_times']:
            for tt in stats_l:
                grasp_t.append(tt)

    # Analyze the statistics globally.
    print("\n================== NOW RELEVANT STATISTICS ==================")

    print("\nTimes for moving to other side, length: {}".format(len(move_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(move_t), np.std(move_t)))

    print("\nTimes for executing grasps, length: {}".format(len(grasp_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(grasp_t), np.std(grasp_t)))

    print("\nTimes for neural net forward pass")
    print("len(grasp): {}".format(len(grasp_net_t)))
    print("len(success): {}".format(len(success_net_t)))

    print("{:.3f}\pm {:.1f}".format(np.mean(grasp_net_t), np.std(grasp_net_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(success_net_t), np.std(success_net_t)))
    combo = np.concatenate((grasp_net_t,success_net_t))
    print("The combined version, with numpy shape {}".format(combo.shape))
    print("{:.3f}\pm {:.1f}".format(np.mean(combo), np.std(combo)))
