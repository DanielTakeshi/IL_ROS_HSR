"""Use this script for inspecting results after doing bed-making deployment.

See the bed-making deployment code for how we saved things.
There are lots of things we can do for inspection.

ALL BUT THE COVERAGE RESULTS.
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
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def analyze_honda():
    """Now do something similar, except for Honda's data style.
    """
    PATHS = sorted(
        [join(RESULTS,x) for x in os.listdir(RESULTS) 
         if 'honda_' in x and 'v01' not in x]
    )
    print("Looking at these paths, which do NOT include Honda's stuff:")
    for item in PATHS:
        print('  '+item)
    print("")

    # Move times, should be pretty slow. Measured for all.
    move_t = []
    grasp_t = []
    net_t = []

    # Record all this together
    num_grasps_all = []
 
    for pth in PATHS:
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
            # WAIT, that's for MY stuff. For their data I only used the two images.

            final_dict = data[-1]
            print("{}".format(rollout_pfile))
            print("data[-2].keys(): {}".format(data[-2].keys()))
            print("data[-1].keys(): {}".format(data[-1].keys()))
            assert 'image_start' in final_dict and 'image_final' in final_dict

            # Go up to the last one, which has the two images I collected.
            # Here, I need to collect a bunch of stuff for them.
            t_grasp = []
            t_frwd = []
            t_tran = []
            print("count, t_grasp, t_fwrd, t_tran")
            for i,datum in enumerate(data[:-1]):
                print("  {}  {:.1f}  {:.4f}  {:.1f}".format(
                        datum['grasp_counts'],
                        datum['t_grasp'],
                        datum['t_fwrd_pass'],
                        datum['t_transition'])
                )

                # In one case the times are the same so don't record.
                if i > 0 and data[i]['grasp_counts'] > data[i-1]['grasp_counts']:
                    time = data[i]['t_grasp'] - data[i-1]['t_grasp']
                    assert time > 0
                    grasp_t.append(time)

                    # Record forward passes.
                    if '_network_' in pth:
                        net_time = data[i]['t_fwrd_pass'] - data[i-1]['t_fwrd_pass']
                        net_t.append(net_time)

            # For the total grasp attempts, I think that's from data[-2].
            num_grasps.append( data[-2]['grasp_counts'] )
            move_t.append( data[-2]['t_transition'] )

        print(np.mean(num_grasps), np.std(num_grasps))
        num_grasps_all.append(num_grasps)

    # Analyze the statistics globally.
    # Can paste this in the appendix so I remember the output.
    print("\n================== NOW RELEVANT STATISTICS ==================")

    print("\nTimes for moving to other side, length: {}".format(len(move_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(move_t), np.std(move_t)))

    print("\nTimes for executing grasps, len {}".format(len(grasp_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(grasp_t), np.std(grasp_t)))

    print("\nTimes for neural net forward pass")
    print("len(net_t): {}".format(len(net_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(net_t), np.std(net_t)))

    print("\nFor number of grasps:")
    for ng,pth in zip(num_grasps_all,PATHS):
        print("{:.1f} \pm {:.1f}  for  {}".format(np.mean(ng), np.std(ng), (pth.split('/'))[-1]))



def analyze_mine():
    PATHS = sorted(
        [join(RESULTS,x) for x in os.listdir(RESULTS) 
         if 'deploy_' in x and 'old' not in x and 'honda' not in x]
    )
    print("Looking at these paths, which do NOT include Honda's stuff:")
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

    # Record all this together
    num_grasps_all = []
 
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
                g_net, s_net = [], []

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
        num_grasps_all.append(num_grasps)

        # Add these to the global lists.
        for stats_l in stats['move_times']:
            for tt in stats_l:
                move_t.append(tt)
        for stats_l in stats['grasp_times']:
            for tt in stats_l:
                grasp_t.append(tt)

    # Analyze the statistics globally.
    # Can paste this in the appendix so I remember the output.
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

    print("\nFor number of grasps:")
    for ng,pth in zip(num_grasps_all,PATHS):
        print("{:.1f} \pm {:.1f}  for  {}".format(np.mean(ng), np.std(ng), (pth.split('/'))[-1]))


if __name__ == "__main__":
    print("Searching in path: {}\n".format(RESULTS))

    print("\n\n === Now my style. ===\n")
    analyze_mine()

    print("\n\n === Now Honda's style. ===\n")
    analyze_honda()
