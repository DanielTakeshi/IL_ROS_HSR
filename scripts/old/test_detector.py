''''
    Used to train a nerual network that maps an image to robot pose (x,y,z)
    Supports the Option of Synthetically Growing the data for Rotation and Translation 

    Author: Michael Laskey 


    FlagsO
    ----------
    --first (-f) : int
        The first roll out to be trained on (required)

    --last (-l) : int
        The last rollout to be trained on (reguired)

    --net_name (-n) : string
        The name of the network if their exists multiple nets for the task (not required)

    --demonstrator (-d) : string 
        The name of the person who trained the network (not required)

'''


import sys, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import IPython
from il_ros_hsr.tensor import inputdata
import numpy as np, argparse
import cPickle as pickle
import cv2
from skvideo.io import vwrite
from il_ros_hsr.core.yolo_detector import YoloDetect


#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM

#specific: fetches specific net file
#from deep_lfd.tensor.nets.net_grasp import Net_Grasp as Net 
########################################################



Options = options()
yolo_detect = YoloDetect()



if __name__ == '__main__':

    user_name = 'corl_anne_n0/'
    Options.setup(Options.root_dir,user_name)

 

    f = []
    for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir): #specific: sup_dir from specific options
        print dirpath
        print filenames
        f.extend(dirnames)

    train_data = []
    test_data = []
    com = COM()

    for filename in f:
        
       
        rollout_data = pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))

        
        for state in rollout_data:

            c_img = state['color_img']
            d_img = state['depth_img']
           
            
           
            yolo_detect.get_detect(c_img,d_img)

       
    
