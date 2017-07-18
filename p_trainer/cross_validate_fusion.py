import cPickle as pickle
import sys, os
import IPython
from compile_sup import Compile_Sup
from il_ros_hsr.tensor import inputdata_f
import numpy as np, argparse
from numpy.random import random
import cv2
import tensorflow as tf
from scipy.misc import imresize
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.vgg_options import VGG_Options as Options
from il_ros_hsr.p_pi.com import Safe_COM as COM
# from il_ros_hsr.p_pi.features import Features
from il_ros_hsr.p_pi.feature_nets.vgg16 import vgg16
from il_ros_hsr.p_pi.feature_nets.pose_estimation import PoseEstimation as PE
#specific: fetches specific net file
from il_ros_hsr.tensor.nets.net_vgg import VggNet as Net_VGG
from il_ros_hsr.tensor.nets.net_pose_estimation import PoseEstimationNet as Net_PE
########################################################

if __name__ == '__main__':
    ITERATIONS = 1000
    BATCH_SIZE = 100
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
    def identity_color(state):
        c_img = state['color_img']
        c_img = imresize(c_img, (224, 224))
        return c_img

    feature_spaces = []
    #VGG
    vgg_f_out = lambda netclass: netclass.pool5_flat
    v_w = 'src/il_ros_hsr/p_pi/safe_corl/vgg/vgg16_weights.npz'
    vgg_dict = {"f_net": vgg16, "f_out": vgg_f_out, "weights": v_w, "o_net": Net_VGG, "sdim": 25088, "run": True, "name": "vgg"}
    feature_spaces.append(vgg_dict)

    #pose estimation
    p_w = 'pytorch_kinematic/pose_weights.npz'
    #pose branch 0
    pose0_f_out = lambda netclass: netclass.blocks_flat["0"]
    pose0_dict = {"f_net": PE, "f_out": pose0_f_out, "weights": p_w, "o_net": Net_PE, "sdim": 100352, "run": True, "name": "pose0"}
    feature_spaces.append(pose0_dict)
    #pose branch1/2
    for step in range(1, 7):
        for branch in range(1, 3):
            ind = str(step) + "_" + str(branch)
            name = "pose" + ind
            sdim = 29792 if branch == 1 else 14896

            pose_f_out = lambda netclass, theInd=ind: netclass.blocks_flat[theInd]
            pose_dict = {"f_net": PE, "f_out": pose_f_out, "weights": p_w, "o_net": Net_PE, "sdim": sdim, "run": True, "name": name}
            feature_spaces.append(pose_dict)

    for feature_space in feature_spaces:
        if feature_space["run"]:
            print("starting " + feature_space["name"])
            data = inputdata_f.IMData(raw_data, state_space=identity_color, precompute=True)

            all_train_losses = []
            all_test_losses = []
            train_times = []
            print("running cross-validation trials for " + feature_space["name"])

            for trial in range(1):
                print("starting trial " + str(trial))
                data.shuffle()

                net_input = tf.placeholder(tf.float32, [None, 224, 224, 3])
                feature_net = feature_space["f_net"](net_input)
                f_out = feature_space["f_out"](feature_net)

                optimize_net = feature_space["o_net"](options, input_x=f_out, state_dim=feature_space["sdim"])

                sess = tf.Session()
                sess.run(tf.initialize_all_variables())
                feature_net.load_weights(feature_space["weights"], sess)

                start = time.time()
                save_path, train_loss, test_loss = optimize_net.optimize(ITERATIONS, data, sess=sess,
                    batch_size=BATCH_SIZE, save=False, feed_in=feature_net.imgs, split_test=True)
                end = time.time()
                train_times.append(end - start)

                all_train_losses.append(train_loss)
                all_test_losses.append(test_loss)

                optimize_net.clean_up()

            avg_train_loss = np.mean(np.array(all_train_losses), axis=0)
            avg_test_loss = np.mean(np.array(all_test_losses), axis=0)
            avg_train_time = np.mean(train_times)

            stat = {}
            stat['type'] = feature_space["name"]
            stat['path'] = save_path
            stat['all_train_time'] = train_times
            stat['avg_train_time'] = avg_train_time
            stat['all_test_loss'] = all_test_losses
            stat['all_train_loss'] = all_train_losses
            stat['avg_test_loss'] = avg_test_loss
            stat['avg_train_loss'] = avg_train_loss

            state_stats.append(stat)


            pickle.dump(state_stats,open(options.stats_dir+'fusion_cross_validate_stats.p','wb'))
