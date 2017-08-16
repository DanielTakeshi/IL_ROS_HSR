import os
import Pyro4
import time
import cPickle as pickle
import IPython
import cv2

sharer = Pyro4.Proxy("PYRONAME:shared.server")

#robot interface
GLOBAL_PATH = "/home/autolab/Workspaces/michael_working/IL_ROS_HSR/"
CANVAS_DIM = 420.0

class Web_Labeler:


	def __init__(self):
		self.count = 0
	def label_image(self,img):
	    global sharer


	    img_path = GLOBAL_PATH+'shared_data/img_'+str(self.count)+'.png'

	    cv2.imwrite(img_path,img)

	    h_,w_,dim = img.shape

	    sharer.set_img(img_path)
	    sharer.set_img_ready(True)

	    print("robot waiting")
	    while not sharer.is_labeled():
	        pass
	    print("robot done")

	    label = sharer.get_label_data()
	    sharer.set_labeled(False)

	    self.count += 1
	    return self.rescale_labels(label,w_,h_)

	def rescale_labels(self,labels,w_,h_):

		if( labels == None):
			return 

		for label in labels['objects']:

			non_scaled = label['box']

			x_min = non_scaled[0]*(w_/CANVAS_DIM)
			y_min = non_scaled[1]*(h_/CANVAS_DIM)

			x_max = non_scaled[2]*(w_/CANVAS_DIM)
			y_max = non_scaled[3]*(h_/CANVAS_DIM)
			

			label['box'] = [x_min, y_min, x_max,y_max]

		

		return labels




