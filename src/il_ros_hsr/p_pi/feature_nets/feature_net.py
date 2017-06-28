import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import IPython
import pickle
import cv2

class FeatureNet():
    def __init__(self):
        self.parameters = []
        #img_mean from vgg- left unchanged
        self.img_mean = [123.68, 116.779, 103.939]

        self.construct_conv_layers()
        self.construct_fc_layers()

    """ zero-mean input """
    def preprocess(self, imgs):
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant(self.img_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = imgs-mean
        return images

    def max_pool(self, input_layer, num, k = 2):
        return tf.nn.max_pool(input_layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name='pool' + str(num))

    """ a single convolution layer """
    def conv2D(self, input_layer, name, dim1, dim2, relu=True, k_size=3, stride=1):
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal([k_size, k_size, dim1, dim2], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(input_layer, kernel, [1, stride, stride, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[dim2], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            if relu:
                output = tf.nn.relu(out, name=scope)
            else:
                output = out
            self.parameters += [kernel, biases]
            return output

    """ a sequence of uninterrupted convolution layers """
    def conv_series(self, input_layer, series_num, dimensions, k_size=3, lastRelu=True):
        curr_layer = input_layer
        layer_num = 1
        for i, (dim1, dim2) in enumerate(dimensions):
            name = "conv" + str(series_num) + "_" + str(layer_num)
            if not lastRelu and i == len(dimensions) - 1:
                curr_layer = self.conv2D(curr_layer, name, dim1, dim2, relu=False, k_size=k_size)
            else:
                curr_layer = self.conv2D(curr_layer, name, dim1, dim2, k_size=k_size)
            layer_num += 1
        return curr_layer


    """ a set of alternating conv_series and pooling layers """
    def conv_block(self, input_layer, series_dimensions):
        curr_layer = input_layer
        for n, dimension in enumerate(series_dimensions):
            series = self.conv_series(curr_layer, n + 1, dimension)
            curr_layer = self.max_pool(series, n + 1)
        return curr_layer

    """ a single fully connected layer """
    def fc(self, input_layer, name, dim1, dim2, relu=True):
        with tf.name_scope(name) as scope:
            fcw = tf.Variable(tf.truncated_normal([dim1, dim2], dtype=tf.float32, stddev=1e-1), name='weights')
            fcb = tf.Variable(tf.constant(1.0, shape=[dim2], dtype=tf.float32), trainable=True, name='biases')
            fcl = tf.nn.bias_add(tf.matmul(input_layer, fcw), fcb)
            if relu:
                fc = tf.nn.relu(fcl)
            else:
                fc = fcl
            self.parameters += [fcw, fcb]
            return fc
