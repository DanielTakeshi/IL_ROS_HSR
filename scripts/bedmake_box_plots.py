"""Use this script for forming the box/bar plots of coverage results.

We should have had another script which computes coverage. This script just
collects the results.

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
RESULTS  = join(HEAD, 'deploy_network')
FIGURES  = join(HEAD, 'figures')
BLACK    = (0, 0, 0)
GREEN    = (0, 255, 0)
RED      = (0, 0, 255)
WHITE    = (255, 255, 255)

# Convert from file name to readable legend label
# TODO: not using this now since we don't have the data ...
RTYPE_TO_NAME = {
    'deploy_network': 'HSR Data Only',
}

# Other matplotlib
tsize = 35
xsize = 30
ysize = 30
tick_size = 25
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


def bar_plot(coverage_per_data):
    """
    See: https://matplotlib.org/gallery/statistics/barchart_demo.html
    """
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(11*ncols,9*nrows), squeeze=False)

    # Two robots but have six experimental conditions for them.
    n_groups = 6
    groups_hsr = ['hsr_human_white',
                  'hsr_anal_white',
                  'hsr_only_white',
                  'hsr_combo_white',
                  'hsr_combo_teal',
                  'hsr_combo_cal',]
    groups_fetch = ['fetch_human_white',
                    'fetch_anal_white',
                    'fetch_only_white',
                    'fetch_combo_white',
                    'fetch_combo_teal',
                    'fetch_combo_cal',]

    means_hsr   = [np.mean(coverage_per_data[x]) for x in groups_hsr]
    std_hsr     = [np.std(coverage_per_data[x]) for x in groups_hsr]
    means_fetch = [np.mean(coverage_per_data[x]) for x in groups_fetch]
    std_fetch   = [np.std(coverage_per_data[x]) for x in groups_fetch]

    index = np.arange(n_groups)

    # For plotting, we need to set ax.bar with `bar_width` offset for second group.
    rects1 = ax[0,0].bar(index, means_hsr, bar_width, alpha=opacity, color='b',
            yerr=std_hsr, error_kw=error_config, label='HSR')
    rects2 = ax[0,0].bar(index+bar_width, means_fetch, bar_width, alpha=opacity, color='r',
            yerr=std_fetch, error_kw=error_config, label='Fetch')

    # Get labeling right.
    ax[0,0].set_xticklabels(('Human', 'Analy', 'Only', 'Comb-W', 'Comb-T', 'Comb-C'))

    # Bells and whistles
    ax[0,0].set_xlabel('Group', fontsize=xsize)
    ax[0,0].set_ylabel('Final Table Top Coverage', fontsize=ysize)
    ax[0,0].set_title('HSR and Fetch Coverage Results', fontsize=tsize)
    ax[0,0].set_xticks(index + bar_width / 2)
    ax[0,0].tick_params(axis='x', labelsize=tick_size)
    ax[0,0].tick_params(axis='y', labelsize=tick_size)
    ax[0,0].set_ylim([0,100]) # it's coverage, makes sense to start from zero
    ax[0,0].legend(loc="best", ncol=1, prop={'size':legend_size})

    plt.tight_layout()
    figname = join(FIGURES, 'plot_bars_coverage_v01.png')
    plt.savefig(figname)
    print("\nJust saved: {}".format(figname))


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Let's manually go through the files. If we want to compute averages, we can
    # do an `os.listdir()` and parse the file names (be CAREFUL with file names!).
    # --------------------------------------------------------------------------
    PATHS = sorted([x for x in os.listdir(FIGURES) if '.png' not in x])
    coverage_per_data = defaultdict(list)

    for result_type in PATHS:
        print("\nresult type: {}".format(result_type))
        pp = join(FIGURES, result_type)
        images = sorted([x for x in os.listdir(pp) if 'end' in x and 'raw' not in x])

        for i_idx,img in enumerate(images):
            print("on image {} (index {})".format(img,i_idx))
            img = img.replace('.png','')
            fsplit = img.split('_')
            coverage = convert( fsplit[-1] )
            print("  coverage: {}".format(coverage))
            coverage_per_data[result_type].append( coverage )

    # add some fake data for now (TODO: have to get these lol).
    coverage_per_data['hsr_human_white']   = [100, 90, 95, 100, 88]
    coverage_per_data['fetch_human_white'] = [100, 90, 95, 100, 99]

    coverage_per_data['hsr_combo_white']   = [99, 98, 91, 90, 88]
    coverage_per_data['fetch_combo_white'] = [88, 90, 85, 90, 91]

    coverage_per_data['hsr_combo_teal']    = [100, 90, 93, 81, 87]
    coverage_per_data['fetch_combo_teal']  = [88, 91, 95, 93, 81]

    coverage_per_data['hsr_combo_cal']     = [90, 80, 93, 91, 97]
    coverage_per_data['fetch_combo_cal']   = [98, 81, 95, 83, 91]

    coverage_per_data['hsr_only_white']    = [80, 80, 83, 81, 94]
    coverage_per_data['fetch_only_white']  = [98, 71, 85, 93, 93]

    coverage_per_data['hsr_anal_white']    = [50, 60, 70, 80, 81]
    coverage_per_data['fetch_anal_white']  = [55, 59, 66, 77, 81]

    bar_plot(coverage_per_data)
