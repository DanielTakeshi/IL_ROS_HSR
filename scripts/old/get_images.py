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

import IPython
from il_ros_hsr.tensor import inputdata
import numpy as np, argparse
import cPickle as pickle
import cv2
import copy
from skvideo.io import vwrite



#######NETWORK FILES TO BE CHANGED#####################
#specific: imports options from specific options file
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as options 
from il_ros_hsr.p_pi.safe_corl.com import Safe_COM as COM

#specific: fetches specific net file
#from deep_lfd.tensor.nets.net_grasp import Net_Grasp as Net 
########################################################



Options = options()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollouts to be used for training")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollouts to be used for training")


    args = parser.parse_args()

    if args.first is not None:
        first = args.first
    else:
        print "please enter a first value with -f"
        sys.exit()

    if args.last is not None:
        last = args.last
    else:
        print "please enter a last value with -l (not inclusive)"
        sys.exit()



   
    f = []
    for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir): #specific: sup_dir from specific options
        print dirpath
        print filenames
        f.extend(dirnames)

    train_data = []
    test_data = []
    com = COM()

    image_data = []
    count = 0
    for filename in f:
       
        rollout_data = pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))

        state = rollout_data[0]
        
        img = state['color_img']

        com.depth_state(state)

        height,width,layers = img.shape
      
        
        
        #video = vwrite(Options.movies_dir+filename+'.mp4',(width,height))
        
        videos_depth = []
        videos_color = []
        videos_binary = []
        print "BEGINING ROLLOUT"
     
        for state in rollout_data:
            
            img_state = {}
            img_state['color_img'] = state['color_img']
            img_state['depth_img'] = state['depth_img']
            img_state['binary_img'] = com.binary_cropped_state(copy.deepcopy(state))
            img_state['gray_img'] = com.gray_state_cv(copy.deepcopy(state))

            cv2.imshow('debug',state['color_img'])
            cv2.waitKey(30)
            image_data.append(img_state)

        count += 1
        if(count > 4):
            break
        
        pickle.dump(image_data,open(Options.movies_dir+'image_data.p','wb'))
    
