import cPickle as pickle
import sys, os
import IPython
from compile_sup import Compile_Sup
from il_ros_hsr.tensor import inputdata_f
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

    raw_data = []
    labels = []

    for filename in f:
        rollout_data = pickle.load(open(options.rollouts_dir+filename+'/rollout.p','r'))

        raw_data.append(rollout_data)
        labels.append(filename)

    state_stats = []
    com = COM()
    features = Features()

    feature_spaces = []
    #VGG
    feature_spaces.append({"feature": features.vgg_extract, "run": True, "name": "vgg", "net": Net_VGG})
    #pose branch 0
    func0 = lambda state: features.pose_extract(state, 0, -1)
    feature_spaces.append({"feature": func0, "run": True, "name": "pose0", "net": Net_Pose_Estimation})
    #pose branch1/2
    for step in range(1, 7):
        for branch in range(1, 3):
            func = lambda state, theBranch=branch, theStep=step: features.pose_extract(state, theBranch, theStep)
            name = "pose" + str(step) + "_" + str(branch)
            if branch == 1:
                feature_spaces.append({"feature": func, "run": True, "name": name, "net": Net_Pose_Estimation, "sdim": 29792})
            elif branch == 2:
                feature_spaces.append({"feature": func, "run": True, "name": name, "net": Net_Pose_Estimation, "sdim": 14896})

    for feature_space in feature_spaces:
        if feature_space["run"]:
            print("precomputing " + feature_space["name"] + " features")
            data = inputdata_f.IMData(raw_data, state_space = feature_space["feature"] ,precompute= True)

            all_train_losses = []
            all_test_losses = []
            print("running cross-validation trials for " + feature_space["name"])
            for trial in range(10):
                data.shuffle()
                if "sdim" in feature_space:
                    net = feature_space["net"](options, state_dim = feature_space["sdim"])
                else:
                    net = feature_space["net"](options)
                save_path, train_loss,test_loss = net.optimize(ITERATIONS,data, batch_size=BATCH_SIZE,save=False)

                all_train_losses.append(train_loss)
                all_test_losses.append(test_loss)

                net.clean_up()

            print("finished cross validation- saving stats")

            avg_train_loss = np.mean(np.array(all_train_losses), axis=0)
            avg_test_loss = np.mean(np.array(all_test_losses), axis=0)

            stat = {}
            stat['type'] = feature_space["name"]
            stat['test_loss'] = avg_test_loss
            stat['train_loss'] = avg_train_loss
            state_stats.append(stat)

            pickle.dump(state_stats,open(options.stats_dir+'all_cross_validate_stats.p','wb'))

    features.clean_up_nets()
