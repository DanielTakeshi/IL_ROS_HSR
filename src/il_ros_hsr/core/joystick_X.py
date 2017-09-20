import IPython
from numpy.random import multivariate_normal
#import listener
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import cPickle as pickle

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
from il_ros_hsr.core.Xbox import XboxController




class JoyStick_X():

    def __init__(self,com=None,inject_noise=False,noise_scale = 1.0):

        self.alpha = noise_scale

        self.xbox = XboxController()
        self.com = com

        self.i_n = inject_noise

        if(self.i_n):
            self.cov_matrix = pickle.load(open(self.com.Options.stats_dir+'cov_matrix.p','rb'))
        
        self.pubTwist = rospy.Publisher('hsrb/command_velocity',Twist,queue_size=1)
        self.record_actions = np.zeros(2)
        self.curr_action = np.zeros(3)
        self.twist_applied = None
    

    def apply_control(self):
        

        control_state = self.xbox.getControllerState()
        noise_action = None


        left_joystick = control_state['left_stick']
        right_joystick = control_state['right_stick']

        d_pad = control_state['d_pad']
        twist = Twist()
        
        self.curr_action = np.array([left_joystick[0],left_joystick[1],right_joystick[1]])


        twist = self.com.format_twist(self.curr_action)

        if(self.i_n and LA.norm(self.curr_action) > 2e-3):
            noise_action = multivariate_normal(self.curr_action,self.alpha*self.cov_matrix)
            twist = self.com.format_twist(noise_action)

        print d_pad

        self.record_actions = np.array([d_pad[0],d_pad[1]])

        self.twist_applied = twist
      
        self.pubTwist.publish(twist)

        return self.curr_action,noise_action,rospy.Time.now()


    def get_current_control(self):
        return self.curr_action

    def get_current_twist(self):
        return self.twist_applied

    def get_record_actions(self):
        return self.record_actions

    def get_record_actions_passive(self):
        control_state = self.xbox.getControllerState()
        d_pad = control_state['d_pad']
        return np.array([d_pad[0],d_pad[1]])


   