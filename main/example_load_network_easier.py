"""Example of how to load in and use a trained network.

This one does NOT require us to painfully reconstruct the full graph or to
infer what we did from a configuration file. It does, though, require us to
know some variable names, but that's not as bad as full reconstruction.
"""
import cv2, sys, os, time, pickle
import numpy as np
np.set_printoptions(suppress=True, precision=4, edgeitems=5)
from os.path import join
import tensorflow as tf


class Test():

    def __init__(self, meta, ckpt):
        """This is how you can load a model in TF without reconstructing it. :-)
        """
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(meta)
        self.saver.restore(self.sess, ckpt)

        # The main requirement is that we need to know what variable names we used.
        # And another complication: by default we ran this in two shots: one for the
        # pretrained weights, and another for the next part. So we have to run these
        # separately as well. And some names are a bit ambiguous ... :-(.
        self.graph = tf.get_default_graph()
        self.names = sorted( [t.name for t in self.graph.as_graph_def().node] )

        #for nn in self.names:
        #    print(nn)
        #sys.exit()

        # The input and output tensors for the pre-trained weights. Note: we don't
        # want the _weights_ of the 26th layer, but the _output_ tensor. Tricky.
        self.images = self.graph.get_tensor_by_name('images:0')
        self.conv26 = self.graph.get_tensor_by_name('yolo/conv_26/leaky_relu:0')

        # The input and output tensors for the fine-tuned, trained portion.
        self.images_1  = self.graph.get_tensor_by_name('images_1:0')
        self.final_out = self.graph.get_tensor_by_name('yolo_1/fc_36/BiasAdd:0')

        # Also need to feed in training_mode:False for dropout ...
        self.training_mode = self.graph.get_tensor_by_name('training_mode:0')

        # AH! There is a problem here. :-( We don't have the meta-data for the pre-trained
        # weights. This means the pretrained stuff all turns to random inits.
        # This thing is all zero:
        self.last_pretrained = self.graph.get_tensor_by_name('yolo/conv_26/biases:0')
        print(self.sess.run(self.last_pretrained))

        # Whereas this thing, example of a variable we actually trained, is non-zero.
        self.trained_variable = self.graph.get_tensor_by_name('yolo/fc_34/biases:0')
        print(self.sess.run(self.trained_variable))

        sys.exit()


    def _process(self, img):
        """We also need to process it in a similar way, unfortunately.
        """
        image_size = 448
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (image_size, image_size))
        assert inputs.shape == (image_size, image_size, 3)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, image_size, image_size, 3))
        return inputs


    def _resize(self, arr):
        """Same as what's in the `configs/bed_grasp_config.py` file.
        """
        xx = np.array([640, 480])
        raw = (0.5 + arr) * xx
        return raw


    def test(self, all_cv_files):
        """Test on all the data in all our cross validation files.
        """
        L2_results = []

        for test_list in all_cv_files:
            with open(test_list, 'r') as f:
                data = pickle.load(f)
            print("loaded test data: {} (length {})".format(test_list, len(data)))

            for idx,item in enumerate(data):
                d_img = np.copy(item['d_img'])
                d_img = self._process(d_img)

                # Run through fixed, pre-trained weights.
                feed = {self.images: d_img}
                tmp_step = self.sess.run(self.conv26, feed)

                # Run trained portion. Need to squeeze to get rid of leading dimension.
                feed = {self.images_1: tmp_step, self.training_mode: False}
                result = np.squeeze( self.sess.run(self.final_out, feed) )

                # That was done using _scaled_ predictions. Now resize.
                result = self._resize(result)

                # Now evaluate prediction ...
                targ = item['pose']
                L2 = np.sqrt( (result[0]-targ[0])**2 + (result[1]-targ[1])**2 )
                print("prediction {} for {}, pixel L2 {:.1f}".format(result, idx, L2))
                L2_results.append(L2)
        
        print("L2s: {:.1f} +/- {:.1f}".format(np.mean(L2_results), np.std(L2_results)))


if __name__ == "__main__":
    # Find the saved Deep Neural Network checkpoint and load it into the `Test` class.
    HEAD = '/nfs/diskstation/seita/bed-make/'
    g_head_name = 'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_cv_False'
    g_ckpt_name = '08_14_18_24_24_save.ckpt-2500'
    meta = join(HEAD, 'grasp/cache_white_v01', g_head_name, g_ckpt_name+'.meta')
    ckpt = join(HEAD, 'grasp/cache_white_v01', g_head_name, g_ckpt_name)
    test = Test(meta, ckpt)

    # Find the _data_ we want, and load it. This is the SAME data it was trained on.
    # So, logically, this is cheating and we will get very, very good performance.
    path = join(HEAD,'cache_white_v01/')
    all_cv_files = [join(path,x) for x in os.listdir(path) if 'cv_' in x]
    test.test(all_cv_files)
