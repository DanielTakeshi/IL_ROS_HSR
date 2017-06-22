import cPickle as pickle
import sys, os
import IPython
from compile_sup import Compile_Sup
from il_ros_hsr.tensor import inputdata
import numpy as np, argparse
from numpy.random import random
import cv2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.vgg_options import VGG_Options as options
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM
from il_ros_hsr.p_pi.safe_corl.features import Features

#specific: fetches specific net file
from il_ros_hsr.tensor.nets.net_ycb_vgg import Net_YCB_VGG as Net_VGG
from il_ros_hsr.tensor.nets.net_ycb_kinematic import Net_YCB_Kinematic as Net_Kinematic
########################################################

ITERATIONS = 2000
BATCH_SIZE = 200

Options = options()

if __name__ == '__main__':
    f = []

    for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir):
        f.extend(dirnames)

    train_data = []
    test_data = []

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

    state_stats = []
    com = COM()
    features = Features()
    which_to_run = [True, True, True, True]
    ind_of_run = 0
    
    ##################################### VGG ORIGINAL #####################################
    if which_to_run[ind_of_run]:
        print("starting vgg original")
        data = inputdata.IMData(train_data, test_data,state_space = features.vgg_extract,precompute= True)
        print("finished precomputing features")
        net = Net_VGG(Options)
        save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE)

        stat = {}
        stat['type'] = 'vgg_original'
        stat['path'] = save_path
        stat['test_loss'] = test_loss
        stat['train_loss'] = train_loss
        state_stats.append(stat)

        net.clean_up()

        pickle.dump(state_stats,open(Options.stats_dir+'vgg_stats.p','wb'))

    ind_of_run += 1
    ##################################### VGG KINEMATIC 0 #####################################
    if which_to_run[ind_of_run]:
        print("starting vgg kinematic 0")
        data = inputdata.IMData(train_data, test_data,state_space = features.vgg_kinematic_pre_extract,precompute= True)
        print("finished precomputing features")
        net = Net_Kinematic(Options, 0)
        save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE)

        stat = {}
        stat['type'] = 'vgg_kinematic0'
        stat['path'] = save_path
        stat['test_loss'] = test_loss
        stat['train_loss'] = train_loss
        state_stats.append(stat)

        net.clean_up()

        pickle.dump(state_stats,open(Options.stats_dir+'vgg_stats.p','wb'))

    ind_of_run += 1
    ##################################### VGG KINEMATIC 1_1 #####################################
    if which_to_run[ind_of_run]:
        print("starting vgg kinematic 1")
        data = inputdata.IMData(train_data, test_data,state_space = features.vgg_kinematic1_extract,precompute= True)
        print("finished precomputing features")
        net = Net_Kinematic(Options, 1)
        save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE)

        stat = {}
        stat['type'] = 'vgg_kinematic1'
        stat['path'] = save_path
        stat['test_loss'] = test_loss
        stat['train_loss'] = train_loss
        state_stats.append(stat)

        net.clean_up()

        pickle.dump(state_stats,open(Options.stats_dir+'vgg_stats.p','wb'))

    ind_of_run += 1
    ##################################### VGG KINEMATIC 1_2 #####################################
    if which_to_run[ind_of_run]:
        print("starting vgg kinematic 2")
        data = inputdata.IMData(train_data, test_data,state_space = features.vgg_kinematic2_extract,precompute= True)
        print("finished precomputing features")
        net = Net_Kinematic(Options, 2)
        save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE)

        stat = {}
        stat['type'] = 'vgg_kinematic2'
        stat['path'] = save_path
        stat['test_loss'] = test_loss
        stat['train_loss'] = train_loss
        state_stats.append(stat)

        net.clean_up()

        pickle.dump(state_stats,open(Options.stats_dir+'vgg_stats.p','wb'))
