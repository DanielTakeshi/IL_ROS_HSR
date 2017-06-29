
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
    people = ['anne_n1']

    for trial in people: 
        user_name = 'corl_'+trial+'/'
        Options.setup(Options.root_dir,user_name)
        data = []

        f = []
        for (dirpath, dirnames, filenames) in os.walk(Options.evaluations_dir): #specific: sup_dir from specific options
            print dirpath
            print filenames

            f.extend(dirnames)

        for filename in f:
            data.append(pickle.load(open(Options.evaluations_dir+filename+'/rollout.p','r')))


        x  = []
        y = []

        success_rate = 0.0
        for traj in data:

            success_state = traj[-1]

            success_rate = success_rate + float(success_rate['success'])
            N = len(traj)
            for i in range(N-1): 
                state = traj[i]
                cv2.imshow('debug',state['color_img'])

        

        print trail 
        print success_rate/(float(len(data)))


