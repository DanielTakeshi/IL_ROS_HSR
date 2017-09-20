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
from il_ros_hsr.sci_kit.learner import Learner
from compile_sup import Compile_Sup 
import numpy as np, argparse
import cPickle as pickle
from numpy.random import random

from sklearn import linear_model


#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.features import Features
########################################################



Options = options()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollouts to be used for training")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollouts to be used for training")


    args = parser.parse_args()

    if args.first is not None:
        first = args.first
    else:
        print "please enter a first value with -f"
        sys.exit()

    if args.last is not None:
        last = args.last
    else:
        print "please enter a last value with -l (not inclusive)"
        sys.exit()



   
    f = []
    for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir): #specific: sup_dir from specific options
        print dirpath
        print filenames
        f.extend(dirnames)

    train_data = []
    test_data = []
    count = 0

    train_labels = []
    test_labels = []
    for filename in f:
        rollout_data = pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))

        if(random() > 0.2):
            train_data.append(rollout_data)
            train_labels.append(filename)
        else: 
            test_data.append(rollout_data)
            test_labels.append(filename)
        # if count == 0:
        #     train_data.append(rollout_data)
        #     count += 1
        # else: 
        #     test_data.append(rollout_data)

    pickle.dump([train_labels,test_labels],open(Options.stats_dir+'/test_train_s.p','wb'))
    state_stats = []
    features = Features()

    clf = linear_model.LinearRegression(n_jobs = -1)
    learner = Learner(features.hog_color,clf)
    learner.add_data(train_data,test_data)
    learner.train_model()
    train_loss,test_loss = learner.get_stats()

    
    stat = {}
    stat['type'] = 'color_hog_linear'
    stat['test_loss'] = test_loss
    stat['train_loss'] = train_loss
    state_stats.append(stat)

    print "TEST LOSS ",test_loss
    print "TRAIN LOSS ",train_loss

    pickle.dump(state_stats,open(Options.stats_dir+'state_trials_data_scikit.p','wb'))
