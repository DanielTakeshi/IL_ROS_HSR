import IPython
from numpy.random import normal
#import listener
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

import numpy as np
import numpy.linalg as LA
from tf import TransformListener





class JoyStick():

    def __init__(self,noise = 0.2):

        self.noise = 0.2

        self.controller = rospy.Subscriber("/hsrb/joy",Joy , self.c_callback)

        self.record_actions = np.zeros(2)
        self.curr_action = np.zeros(3)
        self.twist_applied = None
    

    def c_callback(self,joyData):
        twist = Twist()
        IPython.embed()
        twist.linear.x = joyData.axes[1] #+ 1*normal()
        twist.angular.z = joyData.axes[0] #+ 1*normal()


        if(LA.norm(joyData.axes[0:3]) > 1e-4 ):
            twist.linear.x = joyData.axes[1] #+ self.noise*normal()
            twist.linear.y = joyData.axes[0] #+ self.noise*normal()
            twist.angular.z = joyData.axes[2] + self.noise*normal()
       

        self.curr_action = np.array([joyData.axes[0],joyData.axes[1],joyData.axes[2]])

        self.record_actions = np.array([joyData.axes[14],joyData.axes[15]])

        self.twist_applied = twist


        self.pubTwist.publish(twist)


    def get_current_control(self):
        return self.curr_action

    def get_current_twist(self):
        return self.twist_applied

    def get_record_actions(self):
        return self.record_actions
   