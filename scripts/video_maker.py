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

    for filename in f:
       
        rollout_data = pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))

        state = rollout_data[0]
        
        img = state['color_img']

        com.depth_state(state)

        height,width,layers = img.shape
      
        a = cv2.cv.CV_FOURCC('M','J','P','G')
        
        #video = vwrite(Options.movies_dir+filename+'.mp4',(width,height))
        
        videos_depth = []
        videos_color = []
        videos_binary = []
        print "BEGINING ROLLOUT"
        count = 0
        for state in rollout_data:

            img = state['color_img']
            img = img[:, :, (2, 1, 0)]
            print 'COUNT ',count
            print state['action']
            print state['image_time'] - state['action_time']
            cv2.imwrite('frame_'+str(count)+'.png',img)


            #cv2.waitKey(50)
            
            videos_color.append(img)
            videos_depth.append(com.depth_state_cv(state))
            videos_binary.append(com.binary_cropped_state(state))
            count += 1
    
        vwrite(Options.movies_dir+filename+'_color.mp4',videos_color)
        vwrite(Options.movies_dir+filename+'_depth.mp4',videos_depth)
        vwrite(Options.movies_dir+filename+'_binary.mp4',videos_binary)
       
    
