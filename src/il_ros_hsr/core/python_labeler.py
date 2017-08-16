import os
import Pyro4
import time
import cPickle as pickle
import IPython
import cv2



from online_labeler import QueryLabeler

#robot interface
GLOBAL_PATH = "/home/autolab/Workspaces/michael_working/IL_ROS_HSR/"
CANVAS_DIM = 420.0

class Python_Labeler:


	def __init__(self,cam):
		
		self.cam = cam

	def label_image(self,image = None):
	    
	    ql = QueryLabeler()
	    ql.run(self.cam)

	    data = ql.label_data

	    del ql

	    return data




