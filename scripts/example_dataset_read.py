"""Very similar to `quick_rollout_check.py`.
Confirms my intuition that the pose really matters for the grasp type, but not
so much for the success data type.
"""
import cv2

import IPython
from numpy.random import normal
import time


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

import cPickle as pickle

#latest, 46-49 from rollout_dart

for rnum in range(0, 50):
	path = 'rollouts_dart/rollout_' + str(rnum) + '/rollout.p'
	

	data = pickle.load(open(path,'rb'))
	print(data)
	count = 0
	for datum in data:

		if type(datum) == list: 
			continue

		if datum['type'] == 'grasp':
			pose = datum['pose']
			c_img = datum['c_img']
			# cv2.imwrite('test_stat_data/rollout_' + str(rnum) + '_grasp_'+str(count)+'.png',c_img)

			cv2.imwrite('rollout_'+rnum+'_grasp_'+str(count)+'.png',img)
			
		count += 1

	


