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
RESULTS  = [
        join(HEAD, 'deploy_network'),
]
FIGURES  = join(HEAD, 'figures')
BLACK    = (0, 0, 0)
GREEN    = (0, 255, 0)
RED      = (0, 0, 255)
WHITE    = (255, 255, 255)

# Convert from file name to readable legend label
RTYPE_TO_NAME = {
    'deploy_network': 'HSR Data Only',
}

# Other matplotlib
tsize = 35
xsize = 28
ysize = 28
tick_size = 22
legend_size = 30

# For bar chart in particular
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def convert(s):
    try:
        return float(s)
    except:
        return s


def bar_plot(coverage_hsr):
    """
    See: https://matplotlib.org/gallery/statistics/barchart_demo.html
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
    hsr_avg_start = [np.mean(coverage_hsr[x+'_start']) for x in groups_hsr]
    hsr_std_start = [np.std(coverage_hsr[x+'_start']) for x in groups_hsr]
    hsr_avg_final = [np.mean(coverage_hsr[x+'_final']) for x in groups_hsr]
    hsr_std_final = [np.std(coverage_hsr[x+'_final']) for x in groups_hsr]
    #print(hsr_avg_start)
    #print(hsr_std_start)
    #print(hsr_avg_final)
    #print(hsr_std_final)
    #sys.exit()
    index = np.arange(n_groups)

    # For plotting, we need to set ax.bar with `bar_width` offset for second group.
    rects1 = ax[0,0].bar(index, hsr_avg_final, bar_width, alpha=opacity, color='b',
            yerr=hsr_std_final, error_kw=error_config, label='HSR')
    rects1 = ax[0,0].bar(index, hsr_avg_start, bar_width, alpha=opacity, color='b',
            yerr=hsr_std_start, error_kw=error_config)
    #rects2 = ax[0,0].bar(index+bar_width, means_fetch, bar_width, alpha=opacity, color='r',
    #        yerr=std_fetch, error_kw=error_config, label='Fetch')

    # Get labeling right.
    ax[0,0].set_xticklabels(
        ('Human\n{:.1f} +/- {:.1f}'.format(
                hsr_avg_final[0], hsr_std_final[0]),
         'Analytic\n{:.1f} +/- {:.1f}'.format(
                hsr_avg_final[1], hsr_std_final[1]),
         'Combo-W\n{:.1f} +/- {:.1f}'.format(
                hsr_avg_final[2], hsr_std_final[2]),
         'Combo-C\n{:.1f} +/- {:.1f}'.format(
                hsr_avg_final[3], hsr_std_final[3]),
         'Combo-T\n{:.1f} +/- {:.1f}'.format(
                hsr_avg_final[4], hsr_std_final[4]),
        )
    )

    # Bells and whistles
    ax[0,0].set_xlabel('Initial and Final Coverage Per Group (Mean +/- Std)', fontsize=xsize)
    ax[0,0].set_ylabel('Blanket Coverage', fontsize=ysize)
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


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Let's manually go through the files. If we want to compute averages, we can
    # do an `os.listdir()` and parse the file names (be CAREFUL with file names!).
    # All of this is with the HSR. I don't have the Fetch's data, unfortunately.
    # --------------------------------------------------------------------------
    print("Searching in path: {}".format(FIGURES))
    PATHS = sorted([x for x in os.listdir(FIGURES) if '.png' not in x])
    coverage_hsr = defaultdict(list)

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
            coverage_hsr[result_type+'_start'].append( coverage_s )
            coverage_hsr[result_type+'_final'].append( coverage_f )

    # FAKE DATA FOR NOW, I haven't done the Teal blanket.
    coverage_hsr['deploy_network_teal_start'].append(0.0)
    coverage_hsr['deploy_network_teal_final'].append(0.0)

    # Quick debugging/listing.
    keys = sorted(list(coverage_hsr.keys()))
    print("\nHere are the keys in `coverage_hsr`:\n{}".format(keys))
    print("(All of these are with the HSR)\n")
    for key in keys:
        mean, std = np.mean(coverage_hsr[key]), np.std(coverage_hsr[key])
        print("  coverage[{}], len {}\n({:.2f} \pm {:.1f})  {}".format(key,
                len(coverage_hsr[key]), mean, std, coverage_hsr[key]))
    print("")

    bar_plot(coverage_hsr)
