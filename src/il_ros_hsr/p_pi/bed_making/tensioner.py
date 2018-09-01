#!/usr/bin/python
# -*- coding: utf-8 -*-
import hsrb_interface, rospy, sys, math, tf, tf2_ros, tf2_geometry_msgs, actionlib, cv2
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
from tmc_suction.msg import SuctionControlAction, SuctionControlGoal
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel as PCM
from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
from il_ros_hsr.core.sensors import Gripper_Torque
import il_ros_hsr.p_pi.bed_making.config_bed as cfg
from tf import TransformListener
import numpy as np
import numpy.linalg as LA


class Tensioner(object):

    def __init__(self):
        self.torque = Gripper_Torque()
        self.tl = TransformListener()


    def get_translation(self,direction):
        pose = self.tl.lookupTransform(direction,'hand_palm_link',rospy.Time(0))
        trans = pose[0]
        return trans


    def get_rotation(self,direction):
        pose = self.tl.lookupTransform(direction,'hand_palm_link',rospy.Time(0))
        rot = pose[1]
        return tf.transformations.quaternion_matrix(rot)


    def compute_bed_tension(self, wrench, direction):
        x = wrench.wrench.force.x
        y = wrench.wrench.force.y
        z = wrench.wrench.force.z 
        force = np.array([x,y,z,1.0])
        rot = self.get_rotation(direction)
        force_perp = np.dot(rot,force)
        print("\nInside p_pi/bed_making/tensioner.py, `compute_bed_tension()`:")
        print("CURRENT FORCES: {}".format(force))
        print("FORCE PERP: {}".format(force_perp))
        return force_perp[1]


    def force_pull(self, whole_body, direction):
        """Pull to the target `direction`, hopefully with the sheet!
        
        I think this splits it into several motions, at most `cfg.MAX_PULLS`, so
        this explains the occasional pauses with the HSR's motion. This is *per*
        grasp+pull attempt, so it's different from the max attempts per side,
        which I empirically set at 4. Actually he set his max pulls to 3, but
        because of the delta <= 1.0 (including equality) this is 4 also.

        Move the end-effector, then quickly read the torque data and look at the
        wrench. The end-effector moves according to a _fraction_ that is
        specified by `s = 1-delta`, so I _hope_ if there's too much tension
        after that tiny bit of motion, that we'll exit. But maybe the
        discretization into four smaller pulls is not fine enough?
        """
        count = 0
        is_pulling = False
        t_o = self.get_translation(direction)
        delta = 0.0
        while delta <= 1.0:
            s = 1.0 - delta
            whole_body.move_end_effector_pose(
                    geometry.pose(x=t_o[0]*s, y=t_o[1]*s, z=t_o[2]*s),
                    direction
            )
            wrench = self.torque.read_data()
            norm = np.abs(self.compute_bed_tension(wrench,direction))
            print("FORCE NORM: {}".format(norm))
            if norm > cfg.HIGH_FORCE:
                is_pulling = True
            if is_pulling and norm < cfg.LOW_FORCE:
                break
            if norm > cfg.FORCE_LIMT:
                break
            delta += 1.0/float(cfg.MAX_PULLS)


if __name__=='__main__':
    pass
