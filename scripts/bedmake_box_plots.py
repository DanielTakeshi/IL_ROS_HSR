"""Use this script for forming the box/bar plots of coverage results.

We should have had another script which computes coverage. This script just
collects the results and plots.

I'm thinking, since it can be tricky to describe via a legend we could just write
down groups A, B, C, etc., and in the caption describe what those are.
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
HEAD     = '/nfs/diskstation/seita/bed-make/results/'
FIGURES  = join(HEAD, 'figures')
BLACK    = (0, 0, 0)
GREEN    = (0, 255, 0)
RED      = (0, 0, 255)
WHITE    = (255, 255, 255)

# Other matplotlib
tsize = 35
xsize = 28
ysize = 28
tick_size = 25
legend_size = 28

# For bar chart in particular
bar_width = 0.35
opacity = 0.4
opacity1 = 0.5
opacity2 = 0.8
error_kw = dict(lw=4, capsize=5, capthick=3)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def convert(s):
    try:
        return float(s)
    except:
        return s


def bar_plots(coverage_info):
    """ See: https://matplotlib.org/gallery/statistics/barchart_demo.html
    Actually this is mostly for debugging since I later separate them.
    """
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(11*ncols,9*nrows), squeeze=False)

    # Two robots but have six experimental conditions for them.
    # Update: do five, ignoring the case when it's trained on the network by itself.
    n_groups = 5
    groups_hsr = ['deploy_human',
                  'deploy_analytic',
                  'deploy_network_white',
                  'deploy_network_cal',
                  'deploy_network_teal',]
    groups_fetch = ['fetch_human_white',
                    'fetch_anal_white',
                    'fetch_combo_white',
                    'fetch_combo_teal',
                    'fetch_combo_cal',]

    # Collect all data for the plots.
    hsr_avg_start = [np.mean(coverage_info[x+'_start']) for x in groups_hsr]
    hsr_std_start = [np.std(coverage_info[x+'_start']) for x in groups_hsr]
    hsr_avg_final = [np.mean(coverage_info[x+'_final']) for x in groups_hsr]
    hsr_std_final = [np.std(coverage_info[x+'_final']) for x in groups_hsr]
    index = np.arange(n_groups)

    # For plotting, we need to set ax.bar with `bar_width` offset for second group.
    rects1 = ax[0,0].bar(index, hsr_avg_final, bar_width, alpha=opacity, color='b',
            yerr=hsr_std_final, error_kw=error_kw, label='HSR')
    rects1 = ax[0,0].bar(index, hsr_avg_start, bar_width, alpha=opacity, color='b',
            yerr=hsr_std_start, error_kw=error_kw)
    #rects2 = ax[0,0].bar(index+bar_width, means_fetch, bar_width, alpha=opacity, color='r',
    #        yerr=std_fetch, error_kw=error_config, label='Fetch')

    # Get labeling right.
    ax[0,0].set_xticklabels(
        ('Human\n{:.1f} +/- {:.1f}'.format(   hsr_avg_final[0], hsr_std_final[0]),
         'Analytic\n{:.1f} +/- {:.1f}'.format(hsr_avg_final[1], hsr_std_final[1]),
         'Net-W\n{:.1f} +/- {:.1f}'.format(   hsr_avg_final[2], hsr_std_final[2]),
         'Net-C\n{:.1f} +/- {:.1f}'.format(   hsr_avg_final[3], hsr_std_final[3]),
         'Net-T\n{:.1f} +/- {:.1f}'.format(   hsr_avg_final[4], hsr_std_final[4]),
        )
    )

    # Bells and whistles
    ax[0,0].set_xlabel('Initial and Final Coverage Per Group (Mean +/- Std)', fontsize=xsize)
    ax[0,0].set_ylabel('Coverage', fontsize=ysize)
    ax[0,0].set_title('HSR and Fetch Coverage Results', fontsize=tsize)
    ax[0,0].set_xticks(index + bar_width / 2)
    ax[0,0].tick_params(axis='x', labelsize=tick_size)
    ax[0,0].tick_params(axis='y', labelsize=tick_size)
    ax[0,0].legend(loc="best", ncol=1, prop={'size':legend_size})

    # It's coverage, but most values are high so might not make sense to start from zero.
    ax[0,0].set_ylim([40,100]) # tune this?

    plt.tight_layout()
    figname = join(FIGURES, 'plot_bars_coverage_v01.png')
    plt.savefig(figname)
    print("\nJust saved: {}".format(figname))


def bar_plots_v2(coverage_info):
    """I split into two so that the second one is for transfer.
    """
    # Two robots but have six experimental conditions for them.
    # Update: do five, ignoring the case when it's trained on the network by itself.
    n_groups_1 = 6
    n_groups_2 = 4
    index_1 = np.arange(n_groups_1)
    index_2 = np.arange(n_groups_2)

    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, squeeze=False,
            figsize=(13*ncols,9*nrows),
            gridspec_kw={'width_ratios': [n_groups_1,n_groups_2]},
    )

    # These are all with the white blanket.
    groups_hsr_1 = ['deploy_human',
                    'deploy_analytic',
                    'deploy_network_white',
    ]

    # The Fetch also with the white blanket.
    groups_fetch = ['honda_human',
                    'honda_analytic',
                    'honda_network_white',
    ]

    # The HSR also did some transfer learning. And trained with RGB.
    groups_hsr_2 = ['deploy_network_cal',
                    'deploy_network_cal_rgb',
                    'deploy_network_teal',
                    'deploy_network_teal_rgb',
    ]

    # Collect all data for the plots.
    hsr_avg_start_1 = [np.mean(coverage_info[x+'_start']) for x in groups_hsr_1]
    hsr_std_start_1 = [np.std(coverage_info[x+'_start'])  for x in groups_hsr_1]
    hsr_avg_final_1 = [np.mean(coverage_info[x+'_final']) for x in groups_hsr_1]
    hsr_std_final_1 = [np.std(coverage_info[x+'_final'])  for x in groups_hsr_1]

    fetch_avg_start = [np.mean(coverage_info[x+'_start']) for x in groups_fetch]
    fetch_std_start = [np.std(coverage_info[x+'_start'])  for x in groups_fetch]
    fetch_avg_final = [np.mean(coverage_info[x+'_final']) for x in groups_fetch]
    fetch_std_final = [np.std(coverage_info[x+'_final'])  for x in groups_fetch]

    hsr_avg_start_2 = [np.mean(coverage_info[x+'_start']) for x in groups_hsr_2]
    hsr_std_start_2 = [np.std(coverage_info[x+'_start'])  for x in groups_hsr_2]
    hsr_avg_final_2 = [np.mean(coverage_info[x+'_final']) for x in groups_hsr_2]
    hsr_std_final_2 = [np.std(coverage_info[x+'_final'])  for x in groups_hsr_2]

    # --------------------------------------------------------------------------
    # For plotting, we need to set ax.bar with `bar_width` offset for second
    # group.  Also for the first plot, if we're going to have both HSR and
    # Fetch, we need to arrange them in a desired order, so I do the human then
    # analytic then network, and within those, HSR then Fetch, for all three. A
    # lot of tedious indexing ... double check with the first bar plot.
    # --------------------------------------------------------------------------

    # Use this to encourage grouping of 'humans', 'analytic', and 'network' cases.
    offset = 0.1

    rects1 = ax[0,0].bar(np.array([0, 2, 4]) + offset,
                         hsr_avg_start_1,
                         bar_width,
                         alpha=opacity1,
                         color='blue',
                         yerr=hsr_std_start_1,
                         error_kw=error_kw,
                         label='H-Start')
    rects2 = ax[0,0].bar(np.array([0, 2, 4]) + bar_width + offset,
                         hsr_avg_final_1,
                         bar_width,
                         alpha=opacity2,
                         color='blue',
                         yerr=hsr_std_final_1,
                         error_kw=error_kw,
                         label='H-Final')
    rects3 = ax[0,0].bar(np.array([1, 3, 5]) - offset,
                         fetch_avg_start,
                         bar_width,
                         alpha=opacity1,
                         color='red',
                         yerr=fetch_std_start,
                         error_kw=error_kw,
                         label='F-Start')
    rects4 = ax[0,0].bar(np.array([1, 3, 5]) + bar_width - offset,
                         fetch_avg_final,
                         bar_width,
                         alpha=opacity2,
                         color='red',
                         yerr=fetch_std_final,
                         error_kw=error_kw,
                         label='F-Final')

    # For the transfer learning aspect
    rects8 = ax[0,1].bar(index_2,
                         hsr_avg_start_2,
                         bar_width,
                         alpha=opacity1,
                         color='blue',
                         yerr=hsr_std_start_2,
                         error_kw=error_kw,
                         label='Start')
    rects9 = ax[0,1].bar(index_2+bar_width,
                         hsr_avg_final_2,
                         bar_width,
                         alpha=opacity2,
                         color='blue',
                         yerr=hsr_std_final_2,
                         error_kw=error_kw,
                         label='Final')

    ax[0,0].axhline(y=100, linestyle='--', color='black')
    ax[0,1].axhline(y=100, linestyle='--', color='black')

    # -------------------------------
    # Get labeling of x-axis right!!!
    # -------------------------------

    # For second subplot. The start/end shouldn't matter.
    len1 = len(coverage_info['deploy_human_start'])
    len2 = len(coverage_info['honda_human_start'])
    len3 = len(coverage_info['deploy_analytic_start'])
    len4 = len(coverage_info['honda_analytic_start'])
    len5 = len(coverage_info['deploy_network_white_start'])
    len6 = len(coverage_info['honda_network_white_start'])
    assert len1 == len(coverage_info['deploy_human_final'])
    assert len2 == len(coverage_info['honda_human_final'])
    assert len3 == len(coverage_info['deploy_analytic_final'])
    assert len4 == len(coverage_info['honda_analytic_final'])
    assert len5 == len(coverage_info['deploy_network_white_final'])
    assert len6 == len(coverage_info['honda_network_white_final'])

    ax[0,0].set_xticklabels(
        ('Human\n{:.1f} +/- {:.1f}\n{} Rollouts'.format(    hsr_avg_final_1[0], hsr_std_final_1[0], len1),
         'Human\n{:.1f} +/- {:.1f}\n{} Rollouts'.format(    fetch_avg_final[0], fetch_std_final[0], len2),
         'Analytic\n{:.1f} +/- {:.1f}\n{} Rollouts'.format( hsr_avg_final_1[1], hsr_std_final_1[1], len3),
         'Analytic\n{:.1f} +/- {:.1f}\n{} Rollouts'.format( fetch_avg_final[1], fetch_std_final[1], len4),
         'Net-White\n{:.1f} +/- {:.1f}\n{} Rollouts'.format(hsr_avg_final_1[2], hsr_std_final_1[2], len5),
         'Net-White\n{:.1f} +/- {:.1f}\n{} Rollouts'.format(fetch_avg_final[2], fetch_std_final[2], len6),
        )
    )

    # For second subplot. The start/end shouldn't matter.
    # But these assertions can catch cases if I killed the coverage script
    # without deleting any unpaired starting images.
    len1 = len(coverage_info['deploy_network_cal_start'])
    len2 = len(coverage_info['deploy_network_cal_rgb_start'])
    len3 = len(coverage_info['deploy_network_teal_start'])
    len4 = len(coverage_info['deploy_network_teal_rgb_start'])
    assert len1 == len(coverage_info['deploy_network_cal_final'])
    assert len2 == len(coverage_info['deploy_network_cal_rgb_final'])
    assert len3 == len(coverage_info['deploy_network_teal_final'])
    assert len4 == len(coverage_info['deploy_network_teal_rgb_final'])

    ax[0,1].set_xticklabels(
        ('Net-Cal\n{:.1f} +/- {:.1f}\n{} Rollouts'.format(  hsr_avg_final_2[0], hsr_std_final_2[0], len1),
         'RGB-Cal\n{:.1f} +/- {:.1f}\n{} Rollouts'.format(  hsr_avg_final_2[1], hsr_std_final_2[1], len2),
         'Net-Teal\n{:.1f} +/- {:.1f}\n{} Rollouts'.format( hsr_avg_final_2[2], hsr_std_final_2[2], len3),
         'RGB-Teal\n{:.1f} +/- {:.1f}\n{} Rollouts'.format( hsr_avg_final_2[3], hsr_std_final_2[3], len4),
        )
    )

    # Depends on the groups we have.
    ax[0,0].set_xticks(index_1 + bar_width/2)
    ax[0,1].set_xticks(index_2 + bar_width/2)

    # Bells and whistles
    for i in range(2):
        ax[0,i].set_ylabel('Coverage (Mean +/- Std)', fontsize=ysize)
        # Actually I don't think we need this label since the axis ticks have a lot of info.
        #ax[0,i].set_xlabel('Experimental Condition', fontsize=xsize)
        ax[0,i].tick_params(axis='x', labelsize=tick_size)
        ax[0,i].tick_params(axis='y', labelsize=tick_size)
        ax[0,i].legend(loc="best", ncol=4, prop={'size':legend_size})

        # It's coverage, but most values are high so might not make sense to start from zero.
        ax[0,i].set_ylim([35,109]) # tune this?

        # Doesn't work as I intended
        #ax[0,i].set_yticklabels((40,50,60,70,80,90,100))

    ax[0,0].set_title('HSR and Fetch Coverage Results', fontsize=tsize)
    ax[0,1].set_title('HSR Transfer Coverage Results', fontsize=tsize)

    plt.tight_layout()
    figname = join(FIGURES, 'plot_bars_coverage_v02.png')
    plt.savefig(figname)
    print("\nJust saved: {}".format(figname))


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Let's manually go through the files. If we want to compute averages, we can
    # do an `os.listdir()` and parse the file names (be CAREFUL with file names!).
    # All of this is with the HSR. I don't have the Fetch's data, unfortunately.
    # UPDATE: we have Fetch now!
    # --------------------------------------------------------------------------
    print("Searching in path: {}".format(FIGURES))
    PATHS = sorted([x for x in os.listdir(FIGURES) if '.png' not in x])
    coverage_info = defaultdict(list)

    for result_type in PATHS:
        print("\nresult type: {}".format(result_type))
        pp = join(FIGURES, result_type)
        images_start = sorted([x for x in os.listdir(pp) if 'start' in x and 'raw' not in x])
        images_final = sorted([x for x in os.listdir(pp) if 'end' in x and 'raw' not in x])

        for i_idx,(img_s,img_f) in enumerate(zip(images_start,images_final)):
            print("on (final) image {} (index {})".format(img_f,i_idx))
            img_s = img_s.replace('.png','')
            img_f = img_f.replace('.png','')
            fsplit_s = img_s.split('_')
            fsplit_f = img_f.split('_')
            coverage_s = convert( fsplit_s[-1] )
            coverage_f = convert( fsplit_f[-1] )
            print("  coverage, start -> final: {} -> {}".format(coverage_s, coverage_f))
            coverage_info[result_type+'_start'].append( coverage_s )
            coverage_info[result_type+'_final'].append( coverage_f )

    # Quick debugging/listing.
    keys = sorted(list(coverage_info.keys()))
    print("\nHere are the keys in `coverage_info`:\n{}".format(keys))
    print("(All of these are with the HSR)\n")
    for key in keys:
        mean, std = np.mean(coverage_info[key]), np.std(coverage_info[key])
        print("  coverage[{}], len {}\n({:.2f} \pm {:.1f})  {}".format(key,
                len(coverage_info[key]), mean, std, coverage_info[key]))
    print("")

    bar_plots(coverage_info)
    bar_plots_v2(coverage_info)
