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
# from il_ros_hsr.p_pi.safe_corl.vgg_options import VGG_Options as options
# from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM


ITERATIONS = 1000
BATCH_SIZE = 100


# Options = options()



if __name__ == '__main__':
    temp_s_dir = "/Users/chrispowers/Documents/research/IL_ROS_HSR/data/corl_chris_n0/stats/"

    acomp = pickle.load(open(temp_s_dir+'all_fcompare_cross_validate_stats.p','r'))
    fcomp = pickle.load(open(temp_s_dir+'fusion_cross_validate_stats.p','r'))

    all_stats = pickle.load(open(temp_s_dir+'all_cross_validate_stats.p','r'))
    extra = pickle.load(open(temp_s_dir+'0all_cross_validate_stats.p', 'r'))
    all_stats = [all_stats[0]] + extra + all_stats[1:]
    """
    for each type, records average loss/time with/without fusion
    """
    # with open("time.txt", "w") as time_file:
    #     time_file.write("normal time    fusion time\n")
    #
    #     for j, a_stat in enumerate(acomp):
    #         f_stat = [f for f in fcomp if f['type'] == a_stat['type']][0]
    #         a_time = a_stat['avg_train_time']
    #         f_time = f_stat['avg_train_time']
    #         time_file.write("%8.2f  %8.2f\n" % (a_time, f_time))
    #
    #         a_loss = a_stat['avg_test_loss']
    #         f_loss = f_stat['avg_test_loss']
    #
    #         i = j + 1
    #         fig = plt.figure(i)
    #         ax = fig.add_subplot(i * 100 + i * 10 + i)
    #
    #         ax.plot(a_loss, label=a_stat['type'] + "_normal")
    #         ax.plot(f_loss, label=f_stat['type'] + "_fusion")
    #         lgd = ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    #
    #         fig.savefig(a_stat['type'] + "_comparison.png", bbox_extra_artists = (lgd,), bbox_inches='tight')

    """
    within fusion/non-fusion, graphs each type together for test/train loss
    """
    j = 1
    for stats, type_label in [(all_stats, "normal")]:
        n = len(stats)

        for loss in ["test_loss", "train_loss"]:
            fig = plt.figure(j)
            ax = fig.add_subplot(j * 100 + j * 10 + j)
            color = iter(plt.cm.rainbow(np.linspace(0,1,n)))
            for stat in stats:
                c = next(color)
                ax.plot(stat["avg_" + loss],label=stat['type'], c = c)
                print(stat["avg_train_time"])
            lgd = ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

            fig.savefig(type_label + "_" + loss + ".png", bbox_extra_artists = (lgd,), bbox_inches='tight')
            j += 1
        """ with fusion/non-fusion, record minimum loss reached for ranking """
        test_mins = [(stat['type'], min(stat['avg_test_loss'])) for stat in stats]
        test_mins.sort(key = lambda x: x[1])
        with open(type_label + ".txt", "w") as min_stats:
            min_stats.write("type   min loss\n")
            for cur_type, cur_min in test_mins:
                min_stats.write(cur_type + " %3.5f\n" % cur_min)
