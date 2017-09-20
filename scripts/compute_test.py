
import sys, os

import IPython

import numpy as np, argparse
import cPickle as pickle
from numpy.random import random

import matplotlib.pyplot as plt

import cv2

#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM


ITERATIONS = 200
BATCH_SIZE = 200


Options = options()



if __name__ == '__main__':

    people = ['chris_n0','chris_n1','chris_n2','anne_n0','anne_n1','anne_n2','carolyn_n0','carolyn_n1','carolyn_n2','matt_n0','matt_n1']
    #people = ['chris_n0']

    for trial in people: 
        user_name = 'corl_'+trial+'/'
        Options.setup(Options.root_dir,user_name)
        stats = pickle.load(open(Options.stats_dir+'best_test_loss.p','r'))

        test_loss = []
        for stat in stats:
            test_loss.append(np.min(stat['test_loss']))


        print trial
        print np.mean(test_loss)
        print np.std(test_loss)


