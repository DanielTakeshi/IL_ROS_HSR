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

class NetSix_C(TensorNet):




    def get_acc(self,y_,y_out):
        y_0 = tf.nn.softmax(y_[:,0:5])
        y_1 = tf.nn.softmax(y_[:,5:10])
        y_2 = tf.nn.softmax(y_[:,10:15]) 
        y_3 = tf.nn.softmax(y_[:,15:20])

        y_o_0 = tf.nn.softmax(y_out[:,0:5])
        y_o_1 = tf.nn.softmax(y_out[:,5:10])
        y_o_2 = tf.nn.softmax(y_out[:,10:15]) 
        y_o_3 = tf.nn.softmax(y_out[:,15:20])

        cp_0 = tf.equal(tf.argmax(y_o_0,1), tf.argmax(y_0,1))
        cp_1 = tf.equal(tf.argmax(y_o_1,1), tf.argmax(y_1,1))
        cp_2 = tf.equal(tf.argmax(y_o_2,1), tf.argmax(y_2,1))
        cp_3 = tf.equal(tf.argmax(y_o_3,1), tf.argmax(y_3,1))

        ac_0 = tf.reduce_mean(tf.cast(cp_0, tf.float32))
        ac_1 = tf.reduce_mean(tf.cast(cp_1, tf.float32))
        ac_2 = tf.reduce_mean(tf.cast(cp_2, tf.float32))
        ac_3 = tf.reduce_mean(tf.cast(cp_3, tf.float32))

        return (ac_0+ac_1+ac_2+ac_3)/4.0

    def sf_max_many(self,h_2):
        y_0 = tf.nn.softmax(h_2[:,0:5])
        y_1 = tf.nn.softmax(h_2[:,5:10])
        y_2 = tf.nn.softmax(h_2[:,10:15]) 
        y_3 = tf.nn.softmax(h_2[:,15:20])

        return tf.concat(1,[y_0,y_1,y_2,y_3])

    def __init__(self):
        self.dir = "./net6/"
        self.name = "net6"
        self.channels = 3

        self.x = tf.placeholder('float', shape=[None, 250, 250, self.channels])
        self.y_ = tf.placeholder("float", shape=[None, 20])


        self.w_conv1 = self.weight_variable([11, 11, self.channels, 5])
        self.b_conv1 = self.bias_variable([5])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

     

        conv_num_nodes = self.reduce_shape(self.h_conv1.get_shape())
        fc1_num_nodes = 128
        
        self.w_fc1 = self.weight_variable([conv_num_nodes, fc1_num_nodes])
        # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_conv_flat = tf.reshape(self.h_conv1, [-1, conv_num_nodes])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv_flat, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([fc1_num_nodes, 20])
        self.b_fc2 = self.bias_variable([20])

        self.h_2 = tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2
        self.y_out = self.sf_max_many(self.h_2)
        
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_out), reduction_indices=[1]))

        self.acc = self.get_acc(self.y_,self.y_out)
        self.train_step = tf.train.AdamOptimizer(1e-4)
        self.train = self.train_step.minimize(self.loss)


