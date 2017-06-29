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
from il_ros_hsr.p_pi.vgg_options import VGG_Options as Options
from il_ros_hsr.p_pi.com import Safe_COM as COM
from il_ros_hsr.p_pi.features import Features

#specific: fetches specific net file
from il_ros_hsr.tensor.nets.net_vgg import VggNet as Net_VGG
from il_ros_hsr.tensor.nets.net_pose_estimation import PoseEstimationNet as Net_Pose_Estimation
########################################################

if __name__ == '__main__':
    ITERATIONS = 2000
    BATCH_SIZE = 200
    options = Options()

    f = []

    for (dirpath, dirnames, filenames) in os.walk(options.rollouts_dir):
        f.extend(dirnames)

    train_data = []
    test_data = []

    train_labels = []
    test_labels = []
    for filename in f:
        rollout_data = pickle.load(open(options.rollouts_dir+filename+'/rollout.p','r'))

        if(random() > 0.2):
            train_data.append(rollout_data)
            train_labels.append(filename)
        else:
            test_data.append(rollout_data)
            test_labels.append(filename)

    state_stats = []
    com = COM()
    features = Features()

    feature_spaces = []
    feature_spaces.append({"feature": features.vgg_extract, "run": False, "name": "vgg", "net": Net_VGG})
    feature_spaces.append({"feature": features.pose_0_extract, "run": False, "name": "pose0", "net": Net_Pose_Estimation})
    feature_spaces.append({"feature": features.pose_1_1_extract, "run": False, "name": "pose1_1", "net": Net_Pose_Estimation})
    feature_spaces.append({"feature": features.pose_1_2_extract, "run": False, "name": "pose1_2", "net": Net_Pose_Estimation})
    feature_spaces.append({"feature": features.pose_6_1_extract, "run": True, "name": "pose6_1", "net": Net_Pose_Estimation})

    for feature_space in feature_spaces:
        if feature_space["run"]:
            print("starting " + feature_space["name"] + " features")
            data = inputdata.IMData(train_data, test_data, state_space = feature_space["feature"] ,precompute= True)
            print("finished precomputing features")
            net = feature_space["net"](options)
            save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE)

            stat = {}
            stat['type'] = feature_space["name"]
            stat['path'] = save_path
            stat['test_loss'] = test_loss
            stat['train_loss'] = train_loss
            state_stats.append(stat)

            net.clean_up()

            pickle.dump(state_stats,open(options.stats_dir+'feature_stats.p','wb'))

    features.clean_up_nets()
