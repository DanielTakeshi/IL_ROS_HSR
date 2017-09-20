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

 
    people = ['chris_n0','chris_n1','chris_n2','anne_n0','anne_n1','anne_n2','carolyn_n0','carolyn_n1','carolyn_n2','matt_n0','matt_n1']

    save_path = '/media/autolab/1tb/data/hsr_clutter_rcnn/images/'
    count = 0
    label_count = 0



    for user_name in people:
        Options.setup(Options.root_dir,'corl_'+user_name+'/')

        f = []
        for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir): #specific: sup_dir from specific options
            print dirpath
            print filenames
            f.extend(dirnames)

        print user_name
        print f
        print Options.rollouts_dir
        for filename in f:
         
           
            rollout_data = pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))
            
            for state in rollout_data:

                img = state['color_img']
              


                if(count % 8 == 0):
                    cv2.imwrite(save_path+'frame_'+str(label_count)+'.png',img)
                    label_count += 1
               
                count += 1
        
            
       
    
