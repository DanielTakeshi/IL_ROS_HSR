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
import copy
import sys
from numpy.random import random as rndnp
from il_ros_hsr.p_pi.safe_corl.lighting import get_lighting

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
    
    def __init__(self, data,channels=3,state_space = None,synth = False,precompute = False):


        self.data_tups = data
       

        if(synth):
            self.synth_traj()

        self.train_tups = []
        self.test_tups = []

        self.i = 0
        self.channels = channels

        self.state_space = state_space

    
        self.precompute = precompute

        if(precompute):
            self.pre_compute_features()

    def shuffle(self):
        self.train_tups = []
        self.test_tups = []
        
        for traj in self.data_tups:
            if(rndnp() > 0.2):
                self.train_tups.append(traj)
            else: 
                self.test_tups.append(traj)



    def synth_traj(self):
        aug_train = []
        for traj in self.train_tups:
            aug_traj = []
            for state in traj:
                aug_states = self.synth_color(state)
                for aug_s in aug_states:
                    aug_traj.append(aug_s)

            aug_train.append(aug_traj)

        self.train_tups = aug_train


    def pre_compute_features(self):

        for traj in self.data_tups:
            for data in traj:
                data['feature'] = self.state_space(data)

      
    def synth_color(self,data):

        img = data['color_img']

        img_aug = get_lighting(img)
        states_aug = [data]

        for img in img_aug:
            data_a = copy.deepcopy(data)
         
            data_a['color_img'] = img

            states_aug.append(data_a)

        return states_aug

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
            random.shuffle(traj)
            for data in traj:
                

                action = data['action']

                if(self.precompute):
                    state = data['feature']
                else:
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
        for traj in self.test_tups:
            random.shuffle(traj)
            for data in traj:
                action = data['action']
                
                if(self.precompute):
                    state = data['feature']
                else:
                    state = self.state_space(data)

                
                batch.append((state,action))

        
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])
