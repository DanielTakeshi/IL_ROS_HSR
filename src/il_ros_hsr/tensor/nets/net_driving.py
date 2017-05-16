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
import deep_lfd.tensor.inputdata
import random
from deep_lfd.tensor.tensornet import TensorNet
#from alan.p_grasp_align.options import Grasp_AlignOptions as options
import time
import datetime

class Net_Driving(TensorNet):

    def get_acc(self,y_,y_out):
        cp = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
       
        ac = tf.reduce_mean(tf.cast(cp, tf.float32))
    
        return ac

    def get_prob(self,y_out):
        v = tf.argmax(y_out,1)
        return y_out[:,v]

    def __init__(self,channels=1):
        self.dir = "./net6/"
        self.name = "grasp_net"
        self.channels = channels
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder('float', shape=[None,300,300,self.channels])
            self.y_ = tf.placeholder("float", shape=[None, 5])


            self.w_conv1 = self.weight_variable([7, 7, self.channels, 5])
            self.b_conv1 = self.bias_variable([5])

            self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

            conv_num_nodes = self.reduce_shape(self.h_conv1.get_shape())
            fc1_num_nodes = 120
            
            self.w_fc1 = self.weight_variable([conv_num_nodes, fc1_num_nodes])
            # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
            self.b_fc1 = self.bias_variable([fc1_num_nodes])

            self.h_conv_flat = tf.reshape(self.h_conv1, [-1, conv_num_nodes])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv_flat, self.w_fc1) + self.b_fc1)

            self.w_fc2 = self.weight_variable([fc1_num_nodes, 5])
            self.b_fc2 = self.bias_variable([5])

            
            self.y_out = tf.nn.softmax(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)
            
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_out), reduction_indices=[1]))
            

            self.acc = self.get_acc(self.y_,self.y_out)

            self.train_step = tf.train.MomentumOptimizer(.003, .9)
            self.train = self.train_step.minimize(self.loss)
            self.initializer = tf.initialize_all_variables()
