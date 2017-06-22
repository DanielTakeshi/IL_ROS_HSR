"""
tensorflow version of layers 0, 1_1, and 1_2 from
https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
based on VGG19 implementation from http://www.cs.toronto.edu/~frossard/post/vgg16/
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
# from imagenet_classes import class_names
import IPython
import pickle
import re
import cv2


class vggKin:
    def __init__(self, imgs, weights=None, sess=None, secondBranch=False, weights2=None):
        self.secondBranch = secondBranch
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            if self.secondBranch and weights2 is not None:
                self.load_weights(weights, sess, weights2)
            else:
                self.load_weights(weights, sess)


    def make_conv_layer(self, input_layer, name, dim1, dim2, stride=3):
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal([stride, stride, dim1, dim2], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(input_layer, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[dim2], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            return output


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        self.conv1_1 = self.make_conv_layer(images, "conv1_1", 3, 64)
        self.conv1_2 = self.make_conv_layer(self.conv1_1, "conv1_2", 64, 64)
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        self.conv2_1 = self.make_conv_layer(self.pool1, "conv2_1", 64, 128)
        self.conv2_2 = self.make_conv_layer(self.conv2_1, "conv2_2", 128, 128)
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        self.conv3_1 = self.make_conv_layer(self.pool2, "conv3_1", 128, 256)
        self.conv3_2 = self.make_conv_layer(self.conv3_1, "conv3_2", 256, 256)
        self.conv3_3 = self.make_conv_layer(self.conv3_2, "conv3_3", 256, 256)
        self.conv3_4 = self.make_conv_layer(self.conv3_3, "conv3_4", 256, 256)
        self.pool3 = tf.nn.max_pool(self.conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        self.conv4_1 = self.make_conv_layer(self.pool3, "conv4_1", 256, 512)
        self.conv4_2 = self.make_conv_layer(self.conv4_1, "conv4_2", 512, 512)
        self.conv4_3 = self.make_conv_layer(self.conv4_2, "conv4_3", 512, 256)
        self.conv4_4 = self.make_conv_layer(self.conv4_3, "conv4_4", 256, 128)

        if self.secondBranch > 0:
            self.conv5_1 = self.make_conv_layer(self.conv4_4, "conv5_1", 128, 128)
            self.conv5_2 = self.make_conv_layer(self.conv5_1, "conv5_2", 128, 128)
            self.conv5_3 = self.make_conv_layer(self.conv5_2, "conv5_3", 128, 128)
            self.conv5_4 = self.make_conv_layer(self.conv5_3, "conv5_4", 128, 512, 1)
            lastDim = 38
            if self.secondBranch == 2:
                lastDim = 19
            self.conv5_5 = self.make_conv_layer(self.conv5_4, "conv5_5", 512, lastDim, 1)

    def fc_layers(self):
        in_layer = self.conv4_4
        if self.secondBranch > 0:
            in_layer = self.conv5_5

        # fc1
        with tf.name_scope('fc1') as scope:

            shape = int(np.prod(in_layer.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            self.fc_in_flat = tf.reshape(in_layer, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(self.fc_in_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32), trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess, weight_file2=None):
        #gets 0th layer weights
        def get_key_info(key):
            number = int(re.findall("\.(.*?)\.", key)[0])
            if key[-1] == "t":
                #weight first
                type_num = 0
            else:
                #bias second
                type_num = 1
            return (number, type_num)

        weights = pickle.load( open(weight_file, "rb") )
        keys = sorted(weights.keys(), key = get_key_info)

        if self.secondBranch > 0:
            weights2 = pickle.load( open(weight_file2, "rb") )
            keys2 = sorted(weights2.keys(), key = get_key_info)
            #0th layer comes before branch layer
            keys = keys + keys2
            #combine the dictionaries after sorting
            weights.update(weights2)

        for i, k in enumerate(keys):
            #reverse shape for weights (not biases)
            if len(np.shape(weights[k])) == 4:
                weights[k] = np.swapaxes(weights[k], 0, 3)
                weights[k] = np.swapaxes(weights[k], 1, 2)
            # print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))


if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vggKin(imgs, 'weights.p', sess)


    img1 = imread('testimg.jpg', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.conv4_4_flat, feed_dict={vgg.imgs: [img1]})[0]
    # IPython.embed()
    #use with non-flat version:
    # for i, img in enumerate(prob):
    #     cv2.imwrite("test" + str(i) + ".jpg", img)
