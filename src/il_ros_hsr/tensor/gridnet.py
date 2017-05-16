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

class GridNet(TensorNet):

    def get_acc(self,y_,y_out):
        cp = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
       
        ac = tf.reduce_mean(tf.cast(cp, tf.float32))
    
        return ac

    def get_prob(self,y_out):
        v = tf.argmax(y_out,1)
        return y_out[:,v]

    


    def __init__(self):
        self.dir = "./gridnet/"
        self.name = "gridnet"

        self.x = tf.placeholder('float', shape=[None, 2])
        self.y_ = tf.placeholder("float", shape=[None, 5])


        self.w_fc1 = self.weight_variable([2, 5])
        self.b_fc1 = self.bias_variable([5])

        self.h_1 = tf.nn.relu(tf.matmul(self.x, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([5, 5])
        self.b_fc2 = self.bias_variable([5])

        self.y_out = tf.nn.softmax(tf.matmul(self.h_1, self.w_fc2) + self.b_fc2)
        
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_out), reduction_indices=[1]))
        

        self.acc = self.get_acc(self.y_,self.y_out)
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.loss)


