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


ITERATIONS = 200
BATCH_SIZE = 200


Options = options()



if __name__ == '__main__':



    stats = pickle.load(open(Options.stats_dir+'feature_stats.p','r'))
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for stat in stats:
        ax.plot(stat['test_loss'],label=stat['type'])
        print stat['type']
        print stat['path']

    lgd = ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    
    #plt.show()
    fig.savefig("test_loss.png", bbox_extra_artists = (lgd,), bbox_inches='tight')

    fig = plt.figure(2)
    ax = fig.add_subplot(222)
    for stat in stats:
        ax.plot(stat['train_loss'],label=stat['type'])

    lgd = ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

    fig.savefig("train_loss.png", bbox_extra_artists = (lgd,), bbox_inches='tight')


   
