''''
    Used to train a nerual network that maps an image to robot pose (x,y,z)
    Supports the Option of Synthetically Growing the data for Rotation and Translation

    Author: Michael Laskey


    FlagsO
    ----------
    --first (-f) : int
        The first roll out to be trained on (required)

    --last (-l) : int
        The last rollout to be trained on (reguired)

    --net_name (-n) : string
        The name of the network if their exists multiple nets for the task (not required)

    --demonstrator (-d) : string
        The name of the person who trained the network (not required)

'''


import sys, os

import IPython

import numpy as np, argparse
import cPickle as pickle
from numpy.random import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.vgg_options import VGG_Options as options
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM


ITERATIONS = 1000
BATCH_SIZE = 100


Options = options()



if __name__ == '__main__':

    all_stats = pickle.load(open(Options.stats_dir+'all_cross_validate_stats.p','r'))
    fusion_stats = pickle.load(open(Options.stats_dir+'fusion_cross_validate_stats.p','r'))

    """
    for each type, records average loss/time with/without fusion
    """
    with open("time.txt", "w") as time_file:
        time_file.write("normal time    fusion time\n")

        for i, a_stat in enumerate(all_stats):
            f_stat = [f in fusion_stats if f['type'] == a_stat['type']][0]
            a_time = a_stat['avg_train_time']
            f_time = f_stat['avg_train_time']
            time_file.write("%8.2f  %8.2f\n" % a_time, f_time)

            a_loss = a_stat['avg_test_loss']
            f_loss = f-stat['avg_test_loss']

            fig = plt.figure(i)
            ax = fig.add_subplot(i * 10)

            ax.plot(a_loss, label=a_stat['type'] + "_normal")
            ax.plot(f_loss, label=f_stat['type'] + "_fusion")
            lgd = ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

            fig.savefig(state['type'] + "_comparison.png", bbox_extra_artists = (lgd,), bbox_inches='tight')

    """
    within fusion/non-fusion, graphs each type together for test/train loss
    """
    j = 40
    for stats, type_label in [(all_stats, "normal"), (fusion_stats, "fusion")]:
        n = len(stats)

        for loss in ["test_loss", "train_loss"]:
            fig = plt.figure(j)
            ax = fig.add_subplot(j * 10)
            color = iter(plt.cm.rainbow(np.linspace(0,1,n)))
            for stat in stats:
                c = next(color)
                ax.plot(stat["avg_" + loss],label=stat['type'], c = c)

            lgd = ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

            fig.savefig(type_label + "_" + loss + ".png", bbox_extra_artists = (lgd,), bbox_inches='tight')
            j += 1
        """ with fusion/non-fusion, record minimum loss reached for ranking """
        test_mins = [(stat['type'], min(stat['avg_test_loss'])) for stat in stats]
        test_mins.sort(key = lambda x: x[1])
        with open(type_label + ".txt") as min_stats:
            min_stats.write("type   min loss\n")
            for cur_type, cur_min in test_mins:
                min_stats.write(cur_type + "%3.5f\n" % cur_min)
