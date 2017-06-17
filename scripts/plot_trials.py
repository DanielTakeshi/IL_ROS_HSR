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

import matplotlib.pyplot as plt

#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM


ITERATIONS = 200
BATCH_SIZE = 200


Options = options()



if __name__ == '__main__':

    people = ['chris_n0','chris_n1','chris_n2','anne_n0','anne_n1','anne_n2','carolyn_n0','carolyn_n1','carolyn_n2','matt_n0','matt_n1']


    for trial in people: 
        user_name = 'corl_'+trial+'/'
        Options.setup(Options.root_dir,user_name)

      
            

        stats = pickle.load(open(Options.stats_dir+'trials_final_iteration.p','r'))


        for stat in stats:
            if(not stat['type'] == 'vgg_color_l' ):
                plt.plot(stat['test_loss'],label=trial)
                


        print trial 
        print min(stat['test_loss'])



    plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)


    plt.show()
    IPython.embed()


   