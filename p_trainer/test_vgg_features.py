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

ITERATIONS = 1600
BATCH_SIZE = 200

Options = options()

if __name__ == '__main__':
    f = []
    
    for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir):
        f.extend(dirnames)
    
    train_data = []
    test_data = []
    # count = 0

    # test_cutoff = 5
    # f = [fname for fname in f if int(fname[len("rollout"):]) < test_cutoff]
    
    train_labels = []
    test_labels = []
    for filename in f:
        # count += 1
        # print(count)
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

    ##################################### VGG ORIGINAL #####################################
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
    
    pickle.dump(state_stats,open(Options.stats_dir+'vgg_original.p','wb'))

    ##################################### VGG KINEMATIC #####################################
    print("starting vgg kinematic")
    data = inputdata.IMData(train_data, test_data,state_space = features.vgg_kinematic_extract,precompute= True)
    print("finished precomputing features")
    net = Net_Kinematic(Options)
    save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE)

    stat = {}
    stat['type'] = 'vgg_original'
    stat['path'] = save_path
    stat['test_loss'] = test_loss
    stat['train_loss'] = train_loss
    state_stats.append(stat)

    net.clean_up()

    pickle.dump(state_stats,open(Options.stats_dir+'vgg_kinematic.p','wb'))

    

    

    
