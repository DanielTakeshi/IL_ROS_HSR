import sys
#sys.path.append('/opt/tmc/ros/indigo/lib/python2.7/dist-packages')
import IPython
from numpy.random import multivariate_normal
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import cPickle as pickle
import hsrb_interface
import numpy as np
import numpy.linalg as LA
from tf import TransformListener
from il_ros_hsr.core.Xbox import XboxController




class JoyStick_X():

    def __init__(self):


        self.xbox = XboxController()
        self.robot = hsrb_interface.Robot()
        self.whole_body = self.robot.get('whole_body')
        self.gripper = self.robot.get('gripper')

        self.pubTwist = rospy.Publisher('hsrb/command_velocity',Twist,queue_size=1)
        


    
    def format_twist(self,pos):
        twist = Twist()
        gain = -1.0
        if(np.abs(pos[1]) < 2e-3):
            pos[1] = 0.0

        twist.linear.x = gain*pos[1] #+ self.noise*normal()
        twist.linear.y = gain*pos[0] #+ self.noise*normal()
        twist.angular.z = gain*pos[2] #+ self.noise*normal(

        return twist

    def apply_control(self):
        

        control_state = self.xbox.getControllerState()
        noise_action = None


        left_joystick = control_state['left_stick']
        right_joystick = control_state['right_stick']

        d_pad = control_state['d_pad']
        twist = Twist()

        self.curr_action = np.array([left_joystick[0],left_joystick[1],right_joystick[1]])
   
        twist = self.format_twist(self.curr_action)

        self.record_actions = np.array([d_pad[0],d_pad[1]])

        self.twist_applied = twist
        print twist
        self.pubTwist.publish(twist)

        if(self.record_actions[1] < -0.1):
            print "Move To Home"
            self.gripper.command(1.2)
            self.whole_body.move_to_neutral()

        


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


   
if __name__=='__main__':


    jyX = JoyStick_X()

    while True:

        jyX.apply_control()
