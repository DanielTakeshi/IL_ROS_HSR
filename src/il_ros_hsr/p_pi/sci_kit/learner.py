import tensorflow as tf
from il_ros_hsr.tensor import inputdata
import random
import numpy as np
from il_ros_hsr.tensor.tensornet import TensorNet
#from alan.p_grasp_align.options import Grasp_AlignOptions as options
import time
import datetime
import IPython

import numpy.linalg as LA

class Learner():

	def __init__(self,feature_extractor,model):

		self.features = feature_extractor
		self.model = model


	def add_data(self,train_data,test_data):

		self.X_train = []
		self.Y_train = []
		self.X_test = []
		self.Y_test = []

		for traj in train_data:
			for state in traj:
				self.Y_train.append(state['action'])
				self.X_train.append(self.features(state))

		for traj in test_data:
			for state in traj:
				self.Y_test.append(state['action'])
				self.X_test.append(self.features(state))



	def train_model(self):
		print "TRAINING MODEL"
		self.model.fit(self.X_train,self.Y_train)



	def get_stats(self):

		train_errors = np.zeros(len(self.X_train))
		test_errors = np.zeros(len(self.X_test))

		for i in range(len(self.X_train)):

			y_predict = self.model.predict(self.X_train[i])
			print y_predict
			error = LA.norm(y_predict - self.Y_train[i])
			train_errors[i] = error


		for i in range(len(self.X_test)):

			y_predict = self.model.predict(self.X_test[i])
			error = LA.norm(y_predict - self.Y_test[i])
			test_errors[i] = error

		IPython.embed()
		exp_train = np.sum(train_errors)/float(len(self.X_train))
		exp_test =  np.sum(test_errors)/float(len(self.X_test))


		return exp_train,exp_test

