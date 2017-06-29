"""
    DO NOT INSTANTIATE THIS CLASS!!!!
    Instead use subclasses that actually contain the net architectures

    General information about TensorNet
        - Save will save the current sessions variables to the given path.
          If no path given, it saves to 'self.dir/[timestamped net name].ckpt'
        - Load takes a path and returns a session with those tf variables
        - Optimize will load variables from path if given. Otherwise it will initialize new ones.
          Will save to new timestamp rather than overwriting given path
        - Output takes a session (that was ideally loaded from TensorNet.load) and image and returns
        - the net output in a list. Try not to edit the binary image. output will automatically reformat normal cv2.imread
          or BinaryCamera.read_binary_frame images.

    Try to close sessions after using them (i.e. sess.close()). If more than one is open at a time, exceptions are thrown

    CHANGED save_path directory from self.dir to /media/1tb/Izzy/nets/
"""


import tensorflow as tf
import time
import datetime
import inputdata
import logging
import IPython
#import Analysis from analysis
import numpy as np
import math


import os

class TensorNet():

    def __init__(self):
        self.test_loss = 0
        self.train_loss = 0
        self.Options = options()
        self.analysis = Analysis()
        raise NotImplementedError

    def save(self, sess, save_path=None):
        
        print "Saving..." 
        saver = tf.train.Saver()
        
        model_name = self.name + "_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + ".ckpt"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = save_path + model_name
        
        save_path = saver.save(sess, save_path)
        print "Saved model to "+save_path
        self.recent = save_path
        return save_path


    def clean_up(self):
        self.sess.close()
        tf.reset_default_graph()
        return 

    def load(self, var_path=None):
        """
            load net's variables from absolute path or relative
            to the current working directory. Returns the session
            with those weights/biases restored.
        """
        if not var_path:
            raise Exception("No path to model variables specified")
        print "Restoring existing net from " + var_path + "..."
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            saver.restore(sess, var_path)
        return sess


    def optimize(self, iterations, data, unbiased=False, path=None, batch_size=100, test_print=100, save=True):
        """
            optimize net for [iterations]. path is either absolute or 
            relative to current working directory. data is InputData object (see class for details)
            mini_batch_size as well
        """
        if path:
            self.sess = self.load(var_path=path)
        else:
            print "Initializing new variables..."
            self.sess = tf.Session()
            
            self.sess.run(tf.initialize_all_variables())

        self.test_loss = []
        self.train_loss = []
            
        
        #logging.basicConfig(filename=log_path, level=logging.DEBUG)
        
        try:
            with self.sess.as_default():
                for i in range(iterations):               
                    batch = data.next_train_batch(batch_size)
                    ims, labels = batch
                    
                    feed_dict = { self.x: ims, self.y_: labels }

                    if i % 10 == 0:
                        batch_loss = self.loss.eval(feed_dict=feed_dict)
                        #batch_acc = self.acc.eval(feed_dict=feed_dict)
                    
                        #print "TRAIN: X ERR "+ str(x_loss)+" Y ERR "+str(y_loss)
                        print "[ Iteration " + str(i) + " ] Training loss: " + str(batch_loss)
                        self.train_loss.append(batch_loss)
                        if(math.isnan(batch_loss)):
                            break
                    if i % test_print == 0:
                        test_batch = data.next_test_batch()
                        test_ims, test_labels = test_batch
                        test_dict = { self.x: test_ims, self.y_: test_labels }

                        test_loss = self.loss.eval(feed_dict=test_dict)
                        #test_acc = self.acc.eval(feed_dict=test_dict)
                        
                        #print "TEST: X ERR "+ str(x_loss)+" Y ERR "+str(y_loss)
                        print "[ Iteration " + str(i) + " ] Test loss: " + str(test_loss)
                        self.test_loss.append(test_loss)

                        if(test_loss < 0.025):
                            break

                    self.train.run(feed_dict=feed_dict)
                

        except KeyboardInterrupt:
            pass

        
        
        if save:
            save_path = self.save(self.sess, save_path= self.Options.policies_dir)
        else:
            save_path = None
        #self.sess.close()
        print "Optimization done." 
        return save_path,self.train_loss,self.test_loss
    

    def deploy(self, path, im):
        """
            accepts 3-channel image with pixel values from 
            0-255 and returns controls in four element list
        """
        sess = self.load(var_path=path)
        im = inputdata.im2tensor(im)
        shape = np.shape(im)
        im = np.reshape(im, (-1, shape[0], shape[1], shape[2]))
        with sess.as_default():
            return sess.run(self.y_out, feed_dict={self.x:im})
        
    def predict(self,state):
        #state = state[0]
        A = self.sess.run(self.y_out, feed_dict={self.x:state}) [0]
        return np.argmax(A)#-1
        
    def dist(self,state):
        return self.sess.run(self.y_out, feed_dict={self.x:state}) [0]
    
    def get_stats(self):
        return [self.test_loss, self.train_loss]

    def output(self, sess, im,channels):
        """
            accepts batch of 3d images, converts to tensor
            and returns four element list of controls
        """
        # im = inputdata.im2tensor(im,channels)
      
        shape = im.shape
    
        if(len(shape) == 1):
            im = np.reshape(im, (-1, shape[0]))
        else:
            im = np.reshape(im, (-1, shape[0],shape[1],shape[2]))

        with sess.as_default():
            return sess.run(self.y_out, feed_dict={self.x:im}) [0]



    def class_dist(self,sess,im,channels=3):
        """
        accepts batch of 3d images, converts to tensor
        and returns four element list of controls
        """
        im = inputdata.im2tensor(im,channels)
        shape = np.shape(im)
        im = np.reshape(im, (-1, shape[0], shape[1], shape[2]))
        with sess.as_default():            
            dists = sess.run(self.y_out, feed_dict={self.x:im}) [0]
            return np.reshape(dists, [4,5])


    @staticmethod
    def reduce_shape(shape):
        """
            Given shape iterable, return total number of nodes/elements
        """
        shape = [ x.value for x in shape ]
        f = lambda x, y: 1 if y is None else x * y
        return reduce(f, shape, 1)


    def weight_variable(self, shape, stddev=.005):
        initial = tf.random_normal(shape, stddev=stddev)
        #initial = tf.random_normal(shape)
        #initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape, stddev=.01):
        initial = tf.random_normal(shape, stddev=stddev)
        #initial = tf.random_normal(shape)
        #initial = tf.constant(stddev, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool(self, x, k):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

    def log(self, message):
        print message
        f = open(self.log_path, 'a+')
        #logging.debug(message)
        f.write("DEBUG:root:" + message + "\n")
