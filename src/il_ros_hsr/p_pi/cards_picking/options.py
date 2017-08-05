 #the imports may need to change on actual computer
import os

import cv2
import time
import rospy
import hsrb_interface
from il_ros_hsr.core.options import Options

from il_ros_hsr.core.sensors import RGBD

class Bed_Options(Options):
    # OFFSET_X = 102
    # OFFSET_Y = 150

    # WIDTH = 120 #halve this to look at half of the box with objects
    # HEIGHT = 230 #this currently brings y to the max
    # OFFSET_X = 160
    # OFFSET_Y = 125

    OFFSET_X = 190  # Bin offset
    OFFSET_Y = 200 # Bin offset

    WIDTH = 180  # bin width
    HEIGHT = 180  # bin height

    ROT_MAX = 180
    ROT_MIN = 0.0

    Z_MIN = 0.020
    Z_MAX = 0.080
    # THRESH_TOLERANCE = 80
    # THRESH_TOLERANCE = 50
    THRESH_TOLERANCE = 45

    CHECK_COLLISION = True

    ROT_SCALE = 100.0

    T = 40

    setup_dir = "bed_making/"

    root_dir = "/media/autolab/1tb/"

    def __init__(self):
        
     
        self.setup(self.root_dir, self.setup_dir)






if __name__ == '__main__':

    #hsrb_interface.Robot()
    rospy.init_node('readJoy_node', anonymous=True)
    cam = RGBD()

    c_o = Corl_Options()
    count = 0

    while True: 
        rgb_img = cam.read_color_data()
        count += 1

        if(not rgb_img == None):

            img_cropped = rgb_img[c_o.OFFSET_X:c_o.OFFSET_X+c_o.WIDTH,c_o.OFFSET_Y:c_o.OFFSET_Y+c_o.HEIGHT,:]

            cv2.imshow('debug',img_cropped)

            cv2.waitKey(30) 