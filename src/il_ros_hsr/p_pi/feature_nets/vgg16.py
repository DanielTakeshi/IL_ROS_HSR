########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
"""
updated above implementation with abstraction layers
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
# from imagenet_classes import class_names
import IPython


class vgg16(FeatureNet):
    def __init__(self, imgs, weights=None, sess=None):
        super(PoseEstimation, self).__init__(imgs)

        self.probs = tf.nn.softmax(self.fc3)

        if weights_path is not None and sess is not None:
            self.load_weights(weights_path, sess)

    def construct_conv_layers(self, imgs):
        images = self.preprocess(imgs)
        series_dimensions = []
        series_dimensions.append([(3, 64), (64, 64)])
        series_dimensions.append([(64, 128), (128, 128)])
        series_dimensions.append([(128, 256), (256, 256), (256, 256)])
        series_dimensions.append([(256, 512), (512, 512), (512, 512)])
        series_dimensions.append([(512, 512), (512, 512), (512, 512)])

        self.pool5 = self.conv_branch(images, 5, series_dimensions)

    def construct_fc_layers(self):
        shape = int(np.prod(self.pool5.get_shape()[1:]))
        self.pool5_flat = tf.reshape(self.pool5, [-1, shape])

        self.fc1 = self.fc(self.pool5_flat, "fc1", shape, 4096)
        self.fc2 = self.fc(self.fc1, "fc2", 4096, 4096)
        self.fc3 = self.fc(self.fc2, "fc3", 4096, 1000, relu=False)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            # print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.pool5_flat, feed_dict={vgg.imgs: [img1]})[0]
    IPython.embed()
