import sys, os, time, cv2, argparse
import tty, termios
import numpy as np
import numpy.linalg as LA
import IPython
import rospy
import cPickle as pickle
from il_ros_hsr.core.common import Common
import cv2
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from il_ros_hsr.p_pi.safe_corl.features import Features

###############CHANGGE FOR DIFFERENT PRIMITIVES#########################
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM



class Compute_Noise():


	def __init__(self,user_name = None):
		self.noise = 0.1

		self.start_recording = False
		self.stop_recording = False


		self.com = COM(load_net=True)

		if(not user_name == None):
			self.com.Options.setup(self.com.Options.root_dir,user_name)

		self.com.load_net()

		self.options = self.com.Options

		self.features = Features()




	def get_test_train_split(self):
		train_labels,test_labels = pickle.load(open(self.options.stats_dir+'test_train_f.p','r'))
		self.test_trajs = []

		for filename in test_labels:
			rollout_data =  pickle.load(open(self.options.rollouts_dir+filename+'/rollout.p','r'))

			self.test_trajs.append(rollout_data)


	def compute_covariance_matrix(self):

		self.covariance = np.zeros([3,3])
		self.get_test_train_split()

		N = len(self.test_trajs)

		for traj in self.test_trajs:

			T = float(len(traj))
			t_covariance = np.zeros([3,3])
			for state in traj:
				img = state['color_img']
				action = state['action']

				action_ = self.com.eval_policy(img,self.features.vgg_features, cropped= True)
				action_f = np.zeros([3,1])
				action_f[:,0] = (action-action_)

				rank_one_update = action_f*action_f.T

				#IPython.embed()
				

				t_covariance = t_covariance + rank_one_update

			self.covariance = t_covariance*(1.0/T) + self.covariance


		self.covariance =  self.covariance*(1.0/N)
		print "COMPUTED COVARIANCE MATRIX"
		print self.covariance



	def save_covariance_matrix(self):

		pickle.dump(self.covariance,open(self.options.stats_dir+'cov_matrix.p','wb'))





if __name__ == '__main__':

	username = 'corl_chris_n0/'

	cn = Compute_Noise(username)

	cn.compute_covariance_matrix()
	cn.save_covariance_matrix()

 	







































if __name__ == 'main':

	user_name = 'michael_test'