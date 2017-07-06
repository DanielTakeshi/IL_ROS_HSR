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
    ITERATIONS = 1000
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
    features.clean_up_nets()

    feature_spaces = []
    #VGG
    fvgg = features.vgg
    feature_spaces.append({"f_in": fvgg.imgs, "f_out": fvgg.pool5_flat, "run": True, "name": "vgg", "onet": Net_VGG})
    #pose branch 0
    f_in = features.pose.imgs
    f_blk = features.pose.blocks_flat
    feature_spaces.append({"f_in": f_in, "f_out": f_blk["0"], "run": True, "name": "pose0", "net": Net_Pose_Estimation})
    #pose branch1/2
    for step in range(1, 7):
        for branch in range(1, 3):
            ind = str(step) + "_" + str(branch)
            name = "pose" + ind
            if branch == 1:
                the_sdim = 29792
            elif branch == 2:
                the_sdim = 14896
            feature_spaces.append({"f_in": f_in, "f_out": f_blk[ind], "run": True, "name": name, "net": Net_Pose_Estimation, "sdim": the_sdim})

    for feature_space in feature_spaces:
        if feature_space["run"]:
            print("starting " + feature_space["name"])
            data = inputdata.IMData(train_data, test_data, state_space = lambda x: x ,precompute=True)

            if "sdim" in feature_space:
                net = feature_space["onet"](options, input_x=feature_space["f_out"], state_dim=feature_space["sdim"])
            else:
                net = feature_space["onet"](options, input_x=feature_space["f_out"])

            save_path, train_loss, test_loss = net.optimize(ITERATIONS, data, batch_size=BATCH_SIZE, save=False, feed_in=feature_space["f_in"])

            stat = {}
            stat['type'] = feature_space["name"]
            stat['path'] = save_path
            stat['test_loss'] = test_loss
            stat['train_loss'] = train_loss
            state_stats.append(stat)

            net.clean_up()

            pickle.dump(state_stats,open(options.stats_dir+'fusion_feature_stats.p','wb'))
