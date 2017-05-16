import sys, os, time, cv2, argparse
import tty, termios
import numpy as np
import IPython
import rospy
from il_ros_hsr.core.common import Common
import cv2



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

class Safe_COM(Common):

    @overrides(Common)

    def __init__(self):

        self.Options = Options()
        self.var_path=self.Options.policies_dir+'ycb_05-15-2017_22h50m25s.ckpt'

        self.net = Net(self.Options)
        self.sess = self.net.load(var_path=self.var_path)
        
 
    @overrides(Common)
    def go_to_initial_state(self,whole_body):
        whole_body.move_to_neutral()

        whole_body.move_to_joint_positions({'arm_roll_joint':0.0})
        whole_body.move_to_joint_positions({'wrist_flex_joint':0.0})
        whole_body.move_to_joint_positions({'head_tilt_joint':-0.4})
        whole_body.move_to_joint_positions({'arm_flex_joint':-1.5708})
        whole_body.move_to_joint_positions({'arm_lift_joint':0.23})


    def format_data(self,color_img,depth_img):

        c_img = color_img[self.Options.OFFSET_X:self.Options.OFFSET_X+self.Options.WIDTH,self.Options.OFFSET_Y:self.Options.OFFSET_Y+self.Options.HEIGHT,:]
        d_img = depth_img[self.Options.OFFSET_X:self.Options.OFFSET_X+self.Options.WIDTH,self.Options.OFFSET_Y:self.Options.OFFSET_Y+self.Options.HEIGHT]


        return [c_img, d_img]


    def eval_policy(self,state):
        
        c_img = state[self.Options.OFFSET_X:self.Options.OFFSET_X+self.Options.WIDTH,self.Options.OFFSET_Y:self.Options.OFFSET_Y+self.Options.HEIGHT,:]
        outval = self.net.output(self.sess, c_img,channels=3)
        
        #print "PREDICTED POSE ", pos[2]

        return outval

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


    def depth_state(self,state):
        d_img = state['depth_img']

        d_img[np.where(d_img > 1000)] = 0 
        

        ext_d_img = np.zeros([d_img.shape[0],d_img.shape[1],1])

        ext_d_img[:,:,0] = d_img

        return ext_d_img/1000


    def color_state(self,state):

        return self.im2tensor(state['color_img'])

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




