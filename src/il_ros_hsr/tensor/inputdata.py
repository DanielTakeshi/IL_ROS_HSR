'''
Class to handle test and training data for the neural network

Author : Jonathan Lee

'''

import random
import numpy as np
import numpy.linalg as LA
import IPython
import tensorflow as tf
import cv2
import IPython
import sys


def process_out(n):
    '''
    Computes argmax of a numpy array

    Parameters
    ----------
    n : numpy array

    Returns 
    -------
    out: int
    '''
    out = np.argmax(n)

    return out


def im2tensor(im,channels=1):
    """
    convert 3d image (height, width, 3-channel) where values range [0,255]
    to appropriate pipeline shape and values of either 0 or 1
    cv2 --> tf

    Prameters
    ---------
    im : numpy array 
        matrix with shape of image

    channels : int
        number of channels into the network (Default 1)

    Returns
    -------
    numpy array
        image converted to the correct tensor shape
    """
    shape = np.shape(im)
    h, w = shape[0], shape[1]
    zeros = np.zeros((h, w, channels))
    for i in range(channels):
        zeros[:,:,i] = im[:,:,i]/255.0
    return zeros


class IMData():
    
    def __init__(self, train_data, test_data,channels=3,state_space = None):


        self.train_tups = train_data
        self.test_tups = test_data

        self.i = 0
        self.channels = channels

        self.state_space = state_space

        random.shuffle(self.train_tups)
        random.shuffle(self.test_tups)

    def next_train_batch(self, n):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        if self.i + n > len(self.train_tups):
            self.i = 0
            random.shuffle(self.train_tups)
        batch_tups = self.train_tups[self.i:n+self.i]
        batch = []
        for traj in batch_tups:
            for data in traj:
                
                action = data['action']
                state = self.state_space(data)

                if(len(batch) < 100):
                    batch.append((state,action))

        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])


    def next_test_batch(self):
        """
        read into memory on request
        :return: tuple with images in [0], labels in [1]
        """
        batch = []
        for traj in self.test_tups[:3]:
            for data in traj:
                action = data['action']
                state  = self.state_space(data)

                if(len(batch) < 100):
                    batch.append((state,action))

        random.shuffle(self.test_tups)
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])
