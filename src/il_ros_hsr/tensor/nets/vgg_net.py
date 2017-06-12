import sys
sys.path.append('/Users/chrispowers/Desktop/research/IL_ROS_HSR/src')
import tensorflow as tf
from il_ros_hsr.tensor import inputdata
import random
from il_ros_hsr.tensor.tensornet import TensorNet
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as cOptions
import time
import datetime

class Net_VGG(TensorNet):

    def __init__(self, options,channels=3):
        self.dir = "./net6/"
        self.name = "ycb"
        self.channels = channels
        self.Options = options

        self.x = tf.placeholder('float', shape=[None,250,250,self.channels])

        self.y_ = tf.placeholder("float", shape=[None, 1000])

        #CONV LAYERS

        def apply_conv(conv_in, channels_in, channels_out, window_size = 3):
            w_conv = self.weight_variable([window_size, window_size, channels_in, channels_out])
            b_conv = self.bias_variable([channels_out])
            h_conv = tf.nn.relu(tf.nn.bias_add(self.conv2d(conv_in, w_conv), b_conv))
            return h_conv

        self.conv1_1 = apply_conv(self.x, self.channels, 64)
        self.conv1_2 = apply_conv(self.conv1_1, 64, 64)
        self.pool1 = self.max_pool(self.conv1_2, 2)

        self.conv2_1 = apply_conv(self.pool1, 64, 128)
        self.conv2_2 = apply_conv(self.conv2_1, 128, 128)
        self.pool2 = self.max_pool(self.conv2_2, 2)

        self.conv3_1 = apply_conv(self.pool2, 128, 256)
        self.conv3_2 = apply_conv(self.conv3_1, 256, 256)
        self.conv3_3 = apply_conv(self.conv3_2, 256, 256)
        self.pool3 = self.max_pool(self.conv3_3, 2)

        self.conv4_1 = apply_conv(self.pool3, 256, 512)
        self.conv4_2 = apply_conv(self.conv4_1, 512, 512)
        self.conv4_3 = apply_conv(self.conv4_2, 512, 512)
        self.pool4 = self.max_pool(self.conv4_3, 2)

        self.conv5_1 = apply_conv(self.pool4, 512, 512)
        self.conv5_2 = apply_conv(self.conv5_1, 512, 512)
        self.conv5_3 = apply_conv(self.conv5_2, 512, 512)
        self.pool5 = self.max_pool(self.conv5_3, 2)

        #FC LAYERS

        def apply_fc(fc_in, channels_in, channels_out):
            w_fc = self.weight_variable([channels_in, channels_out])
            b_fc = self.bias_variable([channels_out])
            h_fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc_in, w_fc), b_fc))
            return h_fc

        conv_num_nodes = self.reduce_shape(self.pool5.get_shape())
        self.pool5_flat = tf.reshape(self.pool5, [-1, conv_num_nodes])
        self.fc1 = apply_fc(self.pool5_flat, conv_num_nodes, 4096)
        self.fc2 = apply_fc(self.fc1, 4096, 4096)
        self.fc3 = apply_fc(self.fc2, 4096, 1000)

        self.y_out = tf.nn.softmax(self.fc3)

        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))


        self.train_step = tf.train.MomentumOptimizer(.003, .7)
        self.train = self.train_step.minimize(self.loss)

if __name__ == "__main__":
    testnet = Net_VGG(cOptions)
