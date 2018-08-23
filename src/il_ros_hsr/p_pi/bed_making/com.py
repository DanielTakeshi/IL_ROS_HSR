import sys, os, time, cv2, argparse, IPython, rospy, tty, termios
import numpy as np
from il_ros_hsr.core.common import Common
from geometry_msgs.msg import Twist
import cPickle as pickle

###############CHANGGE FOR DIFFERENT PRIMITIVES#########################
from il_ros_hsr.p_pi.bed_making.options import Bed_Options as Options 
from il_ros_hsr.tensor.nets.net_ycb import Net_YCB as Net
from il_ros_hsr.tensor.nets.net_ycb_vgg import Net_YCB_VGG as Net_VGG
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
#########################################################################

#Common decorator that checks for override, else throws AssertionError
def overrides(super_class):
    def overrider(method):
        assert(method.__name__ in dir(super_class))
        return method
    return overrider


class Bed_COM(Common):

    @overrides(Common)
    def __init__(self,load_net = False,features = None):
        self.Options = Options()
        self.var_path='ycb_06-18-2017_18h02m08s.ckpt'
        self.depth_thresh = 1000
        
 
    @overrides(Common)
    def go_to_initial_state(self,whole_body):
        """Daniel: modified this to get it in a better spot to match Fetch.

        This is for the view-mode of being closer.
        """
        whole_body.move_to_go()
        whole_body.move_to_joint_positions({'arm_flex_joint':  -np.pi/16.0})
        whole_body.move_to_joint_positions({'head_pan_joint':  np.pi/2.0})
        whole_body.move_to_joint_positions({'head_tilt_joint': -np.pi/4.0})# -np.pi/36.0})
        whole_body.move_to_joint_positions({'arm_lift_joint':  0.120})
        

    def save_stat(self, recording, target_path):
        """Saves the recording list to the specified file

        Called from the deployment script, for network deployment.
        """
        rollouts = [x for x in os.listdir(target_path) if 'results_rollout' in x]
        count = len(rollouts)
        K = len(recording)
        pkl_name = 'results_rollout_{}_len_{}.p'.format(count, K)
        final_path = os.path.join(target_path, pkl_name)
        with open(final_path, 'w') as f:
            pickle.dump(recording, f)
        print("Just saved to: {} ...".format(final_path))


    # haven't checked anything beyond this point in detal ...

    def format_data(self,color_img,depth_img):
        c_img = color_img[self.Options.OFFSET_X:self.Options.OFFSET_X+self.Options.WIDTH,self.Options.OFFSET_Y:self.Options.OFFSET_Y+self.Options.HEIGHT,:]
        if(not depth_img == None):
            d_img = depth_img[self.Options.OFFSET_X:self.Options.OFFSET_X+self.Options.WIDTH,self.Options.OFFSET_Y:self.Options.OFFSET_Y+self.Options.HEIGHT]
        else: 
            d_img = None
        return c_img, d_img


    def format_twist(self,pos):
        twist = Twist()
        gain = -0.2
        if(np.abs(pos[1]) < 2e-3):
            pos[1] = 0.0
        twist.linear.x = gain*pos[1] #+ self.noise*normal()
        twist.linear.y = gain*pos[0] #+ self.noise*normal()
        twist.angular.z = gain*pos[2] #+ self.noise*normal(
        return twist


    def eval_policy(self,state,features,cropped = False):
        
        if(not cropped):
            state = state[self.Options.OFFSET_X:self.Options.OFFSET_X+self.Options.WIDTH,self.Options.OFFSET_Y:self.Options.OFFSET_Y+self.Options.HEIGHT,:]
       

        outval = self.net.output(self.sess,features(state),channels=1)
        
        #print "PREDICTED POSE ", pos[2]

        return outval

    def load_net(self):
        self.net = Net_VGG(self.Options,channels=3)
        self.sess = self.net.load(var_path=self.Options.policies_dir+self.var_path)

    def clean_up(self):
        self.sess.close()
        self.net.clean_up()


    def im2tensor(self,im,channels=3):
        for i in range(channels):
            im[:,:,i] = im[:,:,i]/255.0
        return im

    def im2tensor_binary(self,im,channels=3):
        
        cutoff = 140
        for i in range(channels):
            im[:,:,i] = np.round(im[:,:,i]/(cutoff * 2.0)) 
        return im

    def process_depth(self,d_img):
        
        d_img.flags.writeable = True
        d_img[np.where(d_img > 500)] = 0 
        

        ext_d_img = np.zeros([d_img.shape[0],d_img.shape[1],1])

        ext_d_img[:,:,0] = d_img


        return ext_d_img/1000.0

    def state_bins(im, bits=2):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = im/255.0
        im = np.ceil(im * 2**bits)/2**bits
        im = im * 255

        return im.astype(np.uint8)

    def depth_state(self,state):
        d_img = state['depth_img']


        d_img[np.where(d_img > self.depth_thresh)] = 0 
        

        ext_d_img = np.zeros([d_img.shape[0],d_img.shape[1],1])

        ext_d_img[:,:,0] = d_img


        return ext_d_img/float(self.depth_thresh)




    def rgbd(self,state):

        color_img = self.color_state(state)
        depth_img = self.depth_state(state)

        rgbd_im = np.zeros([color_img.shape[0],color_img.shape[1],4])

        rgbd_im[:,:,0:3] = color_img
        rgbd_im[:,:,3] = depth_img[:,:,0]

        return rgbd_im

    def depth_state_cv(self,state):
        d_img = np.copy(state['depth_img'])

        d_img[np.where(d_img > self.depth_thresh)] = 0 
        

        ext_d_img = np.zeros([d_img.shape[0],d_img.shape[1],1])

        ext_d_img[:,:,0] = d_img

        
        return ext_d_img*(255.0/float(self.depth_thresh))

    def binary_cropped_state(self,state):

        d_img = state['depth_img']
        c_img = state['color_img']
     


        return self.im2tensor_binary(c_img)*255.0 


    def color_state(self,state):

        return state['color_img']/255.0

    def color_state_sm(self,state):

        img = cv2.pyrDown(state['color_img'])

        return self.im2tensor(img)

    def color_binary_state(self,state):

        return self.im2tensor_binary(state['color_img'])

    def binary_image(self,img):

        return self.im2tensor_binary(img)


    def gray_state(self,state):
        img = state['color_img']
        
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ext_gray = np.zeros([gray_img.shape[0],gray_img.shape[1],1])
        ext_gray[:,:,0] = gray_img
        return ext_gray/255.0

    def gray_state_cv(self,state):
        img = state['color_img']
        
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ext_gray = np.zeros([gray_img.shape[0],gray_img.shape[1],1])
        ext_gray[:,:,0] = gray_img
        return ext_gray


    def next_rollout(self):
        """
        :return: the String name of the next new potential rollout
                (i.e. do not overwrite another rollout)
        """
        i = 0
        
        prefix = cfg.ROLLOUT_PATH
        

        path = prefix + 'rollout_'+str(i) + "/"
        

        while os.path.exists(path):
            i += 1
            path = prefix + 'rollout_'+str(i) + "/"
            
        return 'rollout_' + str(i)

    def next_stat(self):
        """
        :return: the String name of the next new potential rollout
                (i.e. do not overwrite another rollout)
        """
        i = 0
        
        prefix = cfg.STAT_PATH
        

        path = prefix + 'stat_'+str(i) + "/"
        
        
        while os.path.exists(path):
            i += 1
            path = prefix + 'stat_'+str(i) + "/"
            
        return 'stat_' + str(i)


    def save_rollout(self,recording,rollouts=False):
        """  
        Saves the recoring to the specified file

        Paramters
        ---------
        recording: list 
            The recording of the label point shoud be a list of images and labels

        bc: BinaryCamera

        rollouts: bool 
            If True will save to a rollout directory instead of Supervisor (Default False)
        """
        name = self.next_rollout()
        path = cfg.ROLLOUT_PATH + name + '/'
        print("Inside `p_pi/bed_making/com.py`, saving to: {}".format(path))
        os.makedirs(path)
        pickle.dump(recording,open(path+'rollout.p','wb'))
        print "Done saving."


    def joint_force_state(self,state):
        joints = state['joint_positions']
        j_pose = joints.position
        num_joints = len(j_pose)
        gripper_torque = state['gripper_torque']
        data = np.zeros(num_joints+6)
        data[0:num_joints] = j_pose
        data[num_joints] = gripper_torque.wrench.force.x
        data[num_joints+1] = gripper_torque.wrench.force.y
        data[num_joints+2] = gripper_torque.wrench.force.z
        data[num_joints+3] = gripper_torque.wrench.torque.x
        data[num_joints+4] = gripper_torque.wrench.torque.y
        data[num_joints+5] = gripper_torque.wrench.torque.z
        #IPython.embed()
        return data


if __name__ == '__main__':
    cm = Bed_COM()
    ns = cm.next_stat()
    IPython.embed()
