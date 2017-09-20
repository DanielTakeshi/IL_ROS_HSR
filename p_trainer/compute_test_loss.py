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
from il_ros_hsr.tensor import inputdata_f as inputdata
from compile_sup import Compile_Sup 
import numpy as np, argparse
import cPickle as pickle
from numpy.random import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM
from il_ros_hsr.p_pi.safe_corl.features import Features

#specific: fetches specific net file
from il_ros_hsr.tensor.nets.net_ycb import Net_YCB as Net 
from il_ros_hsr.tensor.nets.net_ycb_t import Net_YCB_T as Net_T 
from il_ros_hsr.tensor.nets.net_ycb_s import Net_YCB_S as Net_S 
from il_ros_hsr.tensor.nets.net_ycb_vgg import Net_YCB_VGG as Net_VGG
from il_ros_hsr.tensor.nets.net_ycb_vgg_l import Net_YCB_VGG_L as Net_VGG_L
########################################################

ITERATIONS = 600
BATCH_SIZE = 200
SAMPLE = 10


Options = options()



if __name__ == '__main__':


    people = ['chris_n0','chris_n1','chris_n2','anne_n0','anne_n1','anne_n2','carolyn_n0','carolyn_n1','carolyn_n2','matt_n0','matt_n1']


    for trial in people:

        user_name = 'corl_'+trial+'/'

        Options.setup(Options.root_dir,user_name)

        #Load Data
        f = []
        for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir): #specific: sup_dir from specific options
            print dirpath
            print filenames
            f.extend(dirnames)


        com = COM()
        features = Features()


        data = []
        for filename in f:
            rollout_data = pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))
            data.append(rollout_data)

        net_data = inputdata.IMData(data,state_space = features.vgg_extract,precompute= True) 


        state_stats = []
        for i in range(SAMPLE):
            f = []
        

            train_data = []
            test_data = []
            count = 0

            train_labels = []
            test_labels = []


            if not os.path.isdir(Options.stats_dir):
                os.makedirs(Options.stats_dir)

           

            net_data.shuffle()
           
            # # # # ###########VGG COLOR#####################################
           
            net = Net_VGG(Options)
            save_path, train_loss,test_loss = net.optimize(ITERATIONS,net_data,save=False, batch_size=BATCH_SIZE)

            stat = {}
            stat['person'] = trial
            stat['test_loss'] = test_loss
            stat['train_loss'] = train_loss
            state_stats.append(stat)

            net.clean_up()

        
        pickle.dump(state_stats,open(Options.stats_dir+'best_test_loss.p','wb'))


