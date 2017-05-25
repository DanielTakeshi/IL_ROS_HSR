import sys, os, time, cv2, argparse
import tty, termios
import numpy as np
import IPython
import rospy
from il_ros_hsr.core.common import Common
import cv2
import tensorflow as tf

from il_ros_hsr.p_pi.safe_corl.vgg.vgg16 import vgg16
from scipy.misc import imread, imresize

###############CHANGGE FOR DIFFERENT PRIMITIVES#########################
from il_ros_hsr.p_pi.safe_corl.options import Corl_Options as Options 
from il_ros_hsr.tensor.nets.net_ycb import Net_YCB as Net
#########################################################################

#Common decorator that checks for override, else throws AssertionError
def overrides(super_class):
    def overrider(method):
        assert(method.__name__ in dir(super_class))
        return method
    return overrider

class Features():

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.sess = tf.Session()
        self.vgg = vgg16(imgs, 'src/il_ros_hsr/p_pi/safe_corl/vgg/vgg16_weights.npz', self.sess)

        
    def clean_up_vgg(self):
        self.sess.close()

 
    def im2tensor(self,im,channels=3):
        
        shape = np.shape(im)
        h, w = shape[0], shape[1]
        zeros = np.zeros((h, w, channels))
        for i in range(channels):
            zeros[:,:,i] = im[:,:,i]/255.0
        return zeros

    def im2tensor_binary(self,im,channels=3):
       
        shape = np.shape(im)
        h, w = shape[0], shape[1]
        zeros = np.zeros((h, w, channels))
        for i in range(channels):
            zeros[:,:,i] = np.round(im[:,:,i]/255.0)
        return zeros

    def process_depth(self,d_img):
        
        d_img.flags.writeable = True
        d_img[np.where(d_img > 1000)] = 0 
        

        ext_d_img = np.zeros([d_img.shape[0],d_img.shape[1],1])

        ext_d_img[:,:,0] = d_img


        return ext_d_img/1000.0

    def depth_state(self,state):
        d_img = state['depth_img']

        d_img[np.where(d_img > 1000)] = 0 
        

        ext_d_img = np.zeros([d_img.shape[0],d_img.shape[1],1])

        ext_d_img[:,:,0] = d_img


        return ext_d_img/1000.0

    def depth_state_cv(self,state):
        d_img = np.copy(state['depth_img'])

        d_img[np.where(d_img > 1000)] = 0 
        

        ext_d_img = np.zeros([d_img.shape[0],d_img.shape[1],1])

        ext_d_img[:,:,0] = d_img


        return ext_d_img*(255.0/1000.0)

    def binary_cropped_state(self,state):

        d_img = state['depth_img']
        c_img = state['color_img']
     

        c_img[np.where(d_img > 1000)] = 0

        return self.im2tensor_binary(c_img)*255.0 


    def color_state(self,state):

        return self.im2tensor(state['color_img'])

    def color_state_sm(self,state):

        img = cv2.pyrDown(state['color_img'])

        return self.im2tensor(img)

    def color_binary_state(self,state):

        return self.im2tensor_binary(state['color_img'])

    def gray_state(self,state):
        img = state['color_img']
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ext_gray = np.zeros([gray_img.shape[0],gray_img.shape[1],1])
        ext_gray[:,:,0] = gray_img
        return ext_gray/255.0


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


    def hog_color(self,state):

        c_img = state['color_img']
        hog_ext = self.hog.compute(c_img)
        return hog_ext[:,0]

    def vgg_features(self,state):
        
        c_img = imresize(state, (224, 224))
        vgg_feat = self.sess.run(self.vgg.pool5_flat,feed_dict={self.vgg.imgs: [c_img]})[0]

        return vgg_feat

    def vgg_extract(self,state):

        c_img = state['color_img']
        c_img = imresize(c_img, (224, 224))

        vgg_feat = self.sess.run(self.vgg.pool5_flat,feed_dict={self.vgg.imgs: [c_img]})[0]

        return vgg_feat