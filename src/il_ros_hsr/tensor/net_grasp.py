"""
    Network takes in a image and outputs (x,y,theta,z)
    Model for net3
        conv
        relu
        pool
        conv
        relu
        pool
        fc
        relu
        fc
        tanh
"""

import sys
sys.path.append('/Users/chrispowers/Desktop/research/IL_ROS_HSR/src')
import tensorflow as tf
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as cOptions
import inputdata
import random
from tensornet import TensorNet
#from alan.p_grasp_align.options import Grasp_AlignOptions as options
import time
import datetime

class Net_Grasp(TensorNet):

    def __init__(self,options,channels=1):
        self.dir = "./net6/"
        self.name = "grasp_net"
        self.channels = channels
        self.Options = options

        self.x = tf.placeholder('float', shape=[None,250,250,self.channels])
        self.y_ = tf.placeholder("float", shape=[None, 4])

        num_conv = 5
        self.w_conv1 = self.weight_variable([7, 7, self.channels, num_conv])
        self.b_conv1 = self.bias_variable([num_conv])
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)
        # print(self.h_conv1.get_shape())

        self.h_conv1 = self.max_pool(self.h_conv1, 3)
        # print(self.h_conv1.get_shape())

        self.w_conv2 = self.weight_variable([7, 7, num_conv, num_conv])
        self.b_conv2 = self.bias_variable([num_conv])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.w_conv2) + self.b_conv2)
        # print(self.h_conv2.get_shape())

        self.h_conv2 = self.max_pool(self.h_conv2, 3)
        # print(self.h_conv2.get_shape())

        conv_num_nodes = self.reduce_shape(self.h_conv2.get_shape())
        self.h_conv_flat = tf.reshape(self.h_conv2, [-1, conv_num_nodes])
        # print(self.h_conv_flat.get_shape())

        fc1_num_nodes = 60

        self.w_fc1 = self.weight_variable([conv_num_nodes, fc1_num_nodes])
        # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv_flat, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([fc1_num_nodes, 4])
        self.b_fc2 = self.bias_variable([4])

        self.y_out = tf.tanh(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(.5*tf.sqrt(tf.square(self.y_out - self.y_)))


        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.loss)

if __name__ == '__main__':
    test = Net_Grasp(cOptions)
