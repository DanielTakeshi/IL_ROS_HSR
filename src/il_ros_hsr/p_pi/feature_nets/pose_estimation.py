"""
tensorflow version of https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
structure based on VGG16 implementation from http://www.cs.toronto.edu/~frossard/post/vgg16/
see pytorch folder for numpy weight file and model_struct.txt (extracted from pytorch .pth weights)
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import IPython
import pickle
import re
import cv2
from il_ros_hsr.p_pi.feature_nets.feature_net import FeatureNet

class PoseEstimation(FeatureNet):
    def __init__(self, imgs, weights_path=None, sess=None):
        self.imgs = imgs
        self.block_outs = {}
        super(PoseEstimation, self).__init__()

        self.probs = tf.nn.softmax(self.fc3)

        if weights_path is not None and sess is not None:
            self.load_weights(weights_path, sess)

    """
    constructed according to model_struct.txt
    used padding = same (no padding size in TensorFlow)
    """
    def construct_conv_layers(self):
        images = self.preprocess(self.imgs)

        # BLOCK 0
        series_dimensions = []
        series_dimensions.append([(3, 64), (64, 64)])
        series_dimensions.append([(64, 128), (128, 128)])
        series_dimensions.append([(128, 256), (256, 256), (256, 256), (256, 256)])
        block0_half = self.conv_block(images, series_dimensions)

        conv_set_num = 4
        conv_dimensions = [(256, 512), (512, 512), (512, 256), (256, 128)]
        self.block_outs["0"] = self.conv_series(block0_half, conv_set_num, conv_dimensions)

        # BLOCK 1_1
        conv_set_num += 1
        conv_dimensions = [(128, 128), (128, 128), (128, 128)]
        self.block_outs["1_1_half"] = self.conv_series(self.block_outs["0"], conv_set_num, conv_dimensions)
        conv_dimensions = [(128, 512), (512, 38)]
        conv_set_num += 1
        self.block_outs["1_1"] = self.conv_series(self.block_outs["1_1_half"], conv_set_num, conv_dimensions, k_size=1, lastRelu=False)

        # BLOCK 1_2
        conv_set_num += 1
        conv_dimensions = [(128, 128), (128, 128), (128, 128)]
        self.block_outs["1_2_half"] = self.conv_series(self.block_outs["0"], conv_set_num, conv_dimensions)
        conv_dimensions = [(128, 512), (512, 19)]
        conv_set_num += 1
        self.block_outs["1_2"] = self.conv_series(self.block_outs["1_2_half"], conv_set_num, conv_dimensions, k_size=1, lastRelu=False)

        # BLOCKS x_1 and x_2 for 2 <= x <= 6
        for stage_num in range(2, 7):
            prev_stage = str(stage_num - 1)
            prev1 = self.block_outs[prev_stage + "_1"]
            prev2 = self.block_outs[prev_stage + "_2"]
            orig0 = self.block_outs["0"]
            
            prev_out = tf.concat([prev1, prev2], 3)
            stage_input = tf.concat([orig0, prev_out], 3)
            
            block1 = str(stage_num) + "_1"
            self.block_outs[block1], conv_set_num = self.construct_recurrent_block(stage_input, 1, conv_set_num)

            block2 = str(stage_num) + "_2"
            self.block_outs[block2], conv_set_num = self.construct_recurrent_block(stage_input, 2, conv_set_num)

    """
    specifically for blocks of the form x_1 or x_2 where x > 1
    """
    def construct_recurrent_block(self, input_layer, branch, conv_set_num):
        conv_set_num += 1
        conv_dimensions = [(185, 128), (128, 128), (128, 128), (128, 128), (128, 128)]
        block_half = self.conv_series(input_layer, conv_set_num, conv_dimensions, k_size=7)
        out_dim = 38 if branch == 1 else 19
        conv_dimensions = [(128, 128), (128, out_dim)]
        conv_set_num += 1
        output_layer = self.conv_series(block_half, conv_set_num, conv_dimensions, k_size=1, lastRelu=False)
        return output_layer, conv_set_num

    """
    creates flattened version of all output layers
    gets the output of this network by itself
    not used for feature representations
    """
    def construct_fc_layers(self):
        self.blocks_flat = {}
        for key in self.block_outs:
            curr_out = self.block_outs[key]
            shape = int(np.prod(curr_out.get_shape()[1:]))
            self.blocks_flat[key] = tf.reshape(curr_out, [-1, shape])

        #fc doesn't actually matter here
        self.fc_in = self.blocks_flat["0"]
        shape = int(np.prod(self.block_outs["0"].get_shape()[1:]))
        self.fc1 = self.fc(self.fc_in, "fc1", shape, 4096)
        self.fc2 = self.fc(self.fc1, "fc2", 4096, 4096)
        self.fc3 = self.fc(self.fc2, "fc3", 4096, 1000, relu=False)

    """
    need to be in same order as parameters, i.e. order layers are constructed
    block order is 0, 1_1, 1_2, 2_1, 2_2, etc.
    layer order for each block is sequential
    weight before bias
    """
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        #weird loading because dictionary saved as np array
        weights = weights['arr_0'][()]

        def get_key_info(key):
            k = key.replace("model", "")
            if k == 0:
                block_num1 = block_num2 = -1
            else:
                block_num1 = k[0]
                block_num2 = k[2]
            layer_num = int(re.findall("\.(.*?)\.", key)[0])
            type_num = 0 if key[-1] == "t" else 1
            return (block_num1, block_num2, layer_num, type_num)

        keys = sorted(weights.keys(), key = get_key_info)

        #check this part for correctness
        for i, k in enumerate(keys):
            print(k)
            print(weights[k].shape)
            print(self.parameters[i].shape)
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
