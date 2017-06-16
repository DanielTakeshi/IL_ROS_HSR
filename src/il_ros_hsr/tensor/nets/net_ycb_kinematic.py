"""
    Network takes in a image and outputs (x,y,theta,z)
    Model for net3
        conv
        relu
        fc
        relu
        fc
        tanh
"""


import tensorflow as tf
from il_ros_hsr.tensor import inputdata
import random
from il_ros_hsr.tensor.tensornet import TensorNet
#from alan.p_grasp_align.options import Grasp_AlignOptions as options
import time
import datetime

class Net_YCB_Kinematic(TensorNet):

    def __init__(self, options,channels=3):
        self.dir = "./net6/"
        self.name = "ycb"
        self.channels = channels
        self.Options = options
        self.sess = tf.Session()

        state_dim = 100352

        self.x = tf.placeholder('float', shape=[None,state_dim])
        self.y_ = tf.placeholder("float", shape=[None, 3])

        #fc1_num_nodes = 60
        fc1_num_nodes = 25

        self.w_fc1 = self.weight_variable([state_dim, fc1_num_nodes])
        # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])


        self.h_fc1 = tf.nn.relu(tf.matmul(self.x, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([fc1_num_nodes, 3])
        self.b_fc2 = self.bias_variable([3])

        self.y_out = tf.tanh(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))


        self.train_step = tf.train.MomentumOptimizer(.003, .7)
        self.train = self.train_step.minimize(self.loss)
