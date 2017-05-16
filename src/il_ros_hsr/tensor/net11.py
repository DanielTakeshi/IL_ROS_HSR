import tensorflow as tf
import inputdata
import random
from tensornet import TensorNet
import time
import datetime

class NetEleven(TensorNet):

    def __init__(self):
        self.dir = "./net11/"
        self.name = "net11"
        channels = 3

        self.x = tf.placeholder('float', shape=[None, 250, 250, channels])
        self.y_ = tf.placeholder("float", shape=[None, 4])


        self.w_conv1 = self.weight_variable([11, 11, channels, 5], .01)
        self.b_conv1 = self.bias_variable([5], .1)
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

        self.w_conv2 = self.weight_variable([5, 5, 5, 3], .01)
        self.b_conv2 = self.bias_variable([3], .1)
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.w_conv2) + self.b_conv2)

        self.w_conv3 = self.weight_variable([3, 3, 3, 3], .01)
        self.b_conv3 = self.bias_variable([3], .1)
        self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.w_conv3) + self.b_conv3)


        conv1_num_nodes = self.reduce_shape(self.h_conv3.get_shape())
        fc1_num_nodes = 1024
        
        self.w_fc1 = self.weight_variable([conv1_num_nodes, fc1_num_nodes])
        # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_conv1_flat = tf.reshape(self.h_conv3, [-1, conv1_num_nodes])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv1_flat, self.w_fc1) + self.b_fc1)

        fc2_num_nodes = 256
        self.w_fc2 = self.weight_variable([fc1_num_nodes, fc2_num_nodes], .01)
        self.b_fc2 = self.bias_variable([fc2_num_nodes], .1)
        self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)


        self.w_fc4 = self.weight_variable([fc2_num_nodes, 4], .01)
        self.b_fc4 = self.bias_variable([4], 0.0)

        self.y_out = tf.tanh(tf.matmul(self.h_fc2, self.w_fc4) + self.b_fc4)

        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))
        self.train_step = tf.train.MomentumOptimizer(.0001, .9)
        self.train = self.train_step.minimize(self.loss)
