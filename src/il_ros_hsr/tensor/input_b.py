import random
import numpy as np
import numpy.linalg as LA
import IPython
import tensorflow as tf
import cv2
import IPython
import sys
sys.path.append("../")

from plot_class import Plotter


class IzzyData_B(InputData):
    def __init__(self, train_path, test_path,channels=3):
        self.train_tups = parse(train_path)
        self.test_tups = parse(test_path)
        self.dist = np.zeros(len(self.train_tups))+1.0/len(self.train_tups)

        self.i = 0
        self.channels = channels

        random.shuffle(self.train_tups)
        random.shuffle(self.test_tups)

    def next_train_batch(self, n):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        batch = np.random.choice(len(self.train_tups),p=self.dist,size=n)
        batch_tups = self.states[batch]

        batch = []
        for path, labels in batch_tups:
            im = cv2.imread(path)
         
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])

    def train_data(self): 
        for path, labels in self.train_tups:
            im = cv2.imread(path)
         
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        batch = zip(*batch)
       
        return list(batch[0]), list(batch[1])


    def next_test_batch(self):
        """
        read into memory on request
        :return: tuple with images in [0], labels in [1]
        """
        batch = []
        for path, labels in self.test_tups[:200]:
            im = cv2.imread(path,self.channels)
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        random.shuffle(self.test_tups)
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])

    def update_weights(self,weights):
        self.dist = weights/np.sum(weights)
        


