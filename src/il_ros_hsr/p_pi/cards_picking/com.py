import sys, os, time, cv2, argparse
import tty, termios
import numpy as np
import IPython
import rospy
from il_ros_hsr.core.common import Common
import cv2

from geometry_msgs.msg import Twist


###############CHANGGE FOR DIFFERENT PRIMITIVES#########################
from il_ros_hsr.p_pi.cards_picking.options import Card_Options as Options 
from il_ros_hsr.tensor.nets.net_ycb import Net_YCB as Net
from il_ros_hsr.tensor.nets.net_ycb_vgg import Net_YCB_VGG as Net_VGG
#########################################################################

#Common decorator that checks for override, else throws AssertionError
def overrides(super_class):
    def overrider(method):
        assert(method.__name__ in dir(super_class))
        return method
    return overrider

class Cards_COM(Common):

    @overrides(Common)

    def __init__(self,load_net = False,features = None):

        self.Options = Options()
        self.var_path='ycb_06-18-2017_18h02m08s.ckpt'
        

        self.depth_thresh = 1000
        
 
    @overrides(Common)
    def go_to_initial_state(self,whole_body):
        whole_body.move_to_go()
        whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
        


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




