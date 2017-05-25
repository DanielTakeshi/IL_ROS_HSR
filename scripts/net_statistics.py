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
import numpy as np, argparse
import cPickle as pickle
import cv2
from skvideo.io import vwrite
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from il_ros_hsr.p_pi.safe_corl.features import Features

#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM

#specific: fetches specific net file
#from deep_lfd.tensor.nets.net_grasp import Net_Grasp as Net 
########################################################



Options = options()
com = COM(load_net = True)
features = Features()


def caculate_error(data):

    errors = np.zeros([len(data),3])
    count = 0
    for state in data:

        img = state['color_img']
        action = state['action']

        action_ = com.eval_policy(img,features.vgg_features, cropped= True)

        dif = np.abs(action-action_)
        errors[count,:] = dif
        count += 1

    total = np.sum(errors,axis=0)

    return total/float(len(data))


def calculate_traj_error(data):

    errors = np.zeros([len(data),3])
    count = 0
    for datum in data:
        errors[count,:] = datum
        count += 1

    total = np.sum(errors,axis=0)

    return total/float(len(data))


if __name__ == '__main__':

    train_labels,test_labels = pickle.load(open(Options.stats_dir+'/test_train_vgg_c_sqrt.p','r'))
    traj_errors = []
    for filename in train_labels:
        rollout_data =  pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))

        error = caculate_error(rollout_data)
        traj_errors.append(error)


    print "AVERAGE TRAINING ERROR: "
    print calculate_traj_error(traj_errors)

    traj_errors = []

    for filename in test_labels:
        rollout_data =  pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))

        error = caculate_error(rollout_data)
        traj_errors.append(error)


    print "AVERAGE TESTING ERROR: "
    print calculate_traj_error(traj_errors)
   
 
            
       
    
