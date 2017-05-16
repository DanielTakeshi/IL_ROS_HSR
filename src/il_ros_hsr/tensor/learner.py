from Net.tensor import EndNet6
import numpy as np


class learner():
	'''
	Acts as a abstract class for different kinds of learning algorithms, containing the method
	signatures that are required by a boostNet class
	'''
	
	def __init__(self):
		pass

	def optimize(self, iterations, data, labels, data_weights):
		'''
		Trains the learner
		'''
		pass

	def start(self):
		'''
		Allows the learner to be evaluated (ex. start the session and assign the session to
		a variable)
		'''
		pass

	def predict(self, data):
		'''
		Get the output of the learner on the dataset
		'''
		pass

	def stop(self):
		'''
		end the evaluation session (ex. end the session)
		'''
		pass

	def error(self, data, labels, weights):
		'''
		Get the error of the learner on the given dataset with the following weights
		'''
		pass

	def accuracy(self, data, labels, weights):
		'''
		Gets the classification rate of the learner on teh given dataset with teh following
		rates
		'''
		pass

class LinNet6Learner(learner):

	def __init__(self, path):
		self.net = EndNet6.EndNet6()
		self.net_path = path
		self.sess = None

	def optimize(self, iterations, data, data_weights):
		self.net_path = self.net.optimize(iterations, data, path = self.conv_path, batch_size=200)

	def start(self):
		self.sess = self.net.load(self.net_path)

	def stop(self):
		self.sess.close()

	def predict(self, data):
		return sess.run(self.net.y_out, feed_dict={self.net.x:data})

	def error(self, data, labels, weights):
		# return sess.run(self.net.)

	def accuracy(self, data, labels, weights)




