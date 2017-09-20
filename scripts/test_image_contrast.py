import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import IPython
import sys
sys.path.append('/home/autolab/Workspaces/michael_working/yolo_tensorflow/')
import configs.config_bed as cfg
import cv2



if __name__ == "__main__":


	#-----Reading the image-----------------------------------------------------
	img = cv2.imread(cfg.IMAGE_PATH+'frame_0.png', 1)

	cv2.imshow("img",img) 
	cv2.waitKey(30)
	#-----Converting image to LAB Color model----------------------------------- 
	lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	# cv2.imshow("lab",lab)
	# cv2.waitKey(30)

	#-----Splitting the LAB image to different channels-------------------------
	l, a, b = cv2.split(lab)
	# cv2.imshow('l_channel', l)
	# cv2.waitKey(30)
	# cv2.imshow('a_channel', a)
	# cv2.waitKey(30)
	# cv2.imshow('b_channel', b)
	# cv2.waitKey(30)
	#-----Applying CLAHE to L-channel-------------------------------------------
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)
#	cv2.imshow('CLAHE output', cl)
#	cv2.waitKey(30)
	#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
	limg = cv2.merge((cl,a,b))
#	cv2.imshow('limg', limg)
#	cv2.waitKey(30)
	#-----Converting image from LAB Color model to RGB model--------------------
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	cv2.imshow('final', final)
	cv2.waitKey(30)

	IPython.embed()