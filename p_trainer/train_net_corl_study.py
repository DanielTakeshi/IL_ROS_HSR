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
from il_ros_hsr.tensor import inputdata
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


Options = options()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollouts to be used for training")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollouts to be used for training")

    parser.add_argument("-u", "--username", type=str,
                        help="enter the name of the person in the study")


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

    if args.username is not None:
        user_name = args.username
    else:
        print "please enter a username"
        sys.exit()


    Options.setup(Options.root_dir,user_name)

    #Load Data
   
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
        

    #############DEBUG##########################
    # train_data.append(pickle.load(open(Options.rollouts_dir+f[0]+'/rollout.p','r')))
    # test_data.append(pickle.load(open(Options.rollouts_dir+f[1]+'/rollout.p','r')))

    if not os.path.isdir(Options.stats_dir):
        os.makedirs(Options.stats_dir)

    pickle.dump([train_labels,test_labels],open(Options.stats_dir+'test_train_f.p','wb'))
    state_stats = []
    com = COM()
    features = Features()


    ###############NO SYNTHETIC###############################


    # # # # ###########VGG COLOR#####################################
    data = inputdata.IMData(train_data, test_data,state_space = features.vgg_extract,precompute= True) 
    net = Net_VGG(Options)
    save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE)

    stat = {}
    stat['type'] = 'vgg_color'
    print stat['type']
    stat['path'] = save_path
    stat['test_loss'] = test_loss
    stat['train_loss'] = train_loss
    state_stats.append(stat)

    net.clean_up()

    pickle.dump(state_stats,open(Options.stats_dir+'trials_final_iteration.p','wb'))


