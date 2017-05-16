"""
    Model for net3
        conv
        relu
        fc
        relu
        fc
        tanh
"""


import tensorflow as tf
import inputdata
import random
from tensornet import TensorNet
import time
import datetime

class NetSLV(TensorNet):

    def __init__(self):
        self.dir = "./net6/"
        self.name = "net6"
        self.channels = 1

        self.x = tf.placeholder('float', shape=[None, 250, 250, self.channels])
        self.y_ = tf.placeholder("float", shape=[5, 4])


        self.w_conv1 = self.weight_variable([11, 11, self.channels, 5])
        self.b_conv1 = self.bias_variable([5])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

        # 280: Max Pooling
        #self.h_conv1 = self.max_pool(self.h_conv1, 4)

        # 280: Add 2nd convolutional layer
        #self.w_conv2 = self.weight_variable([5, 5, 5, 3])
        #self.b_conv2 = self.bias_variable([3])

        #self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.w_conv2) + self.b_conv2)
        #self.h_conv2 = self.max_pool(self.h_conv2, 4)

        # print self.h_conv1.get_shape()
        # conv_num_nodes = self.reduce_shape(self.h_conv2.get_shape())
        # fc1_num_nodes = 128
        
        # self.w_fc1 = self.weight_variable([conv_num_nodes, fc1_num_nodes])
        # # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        # self.b_fc1 = self.bias_variable([fc1_num_nodes])

        # self.h_conv_flat = tf.reshape(self.h_conv2, [-1, conv_num_nodes])

        conv_num_nodes = self.reduce_shape(self.h_conv1.get_shape())
        fc1_num_nodes = 128
        
        self.w_fc1 = self.weight_variable([conv_num_nodes, fc1_num_nodes])
        # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_conv_flat = tf.reshape(self.h_conv1, [-1, conv_num_nodes])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv_flat, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([fc1_num_nodes, 4])
        self.b_fc2 = self.bias_variable([4])

        self.y_out = tf.tanh(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))



        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.loss)


