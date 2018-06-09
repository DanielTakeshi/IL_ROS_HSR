"""
Common Class for Neural Network WayPoint Policies 
Author: Michael Laskey 
"""
import sys, os, time, cv2, argparse
import tty, termios
import numpy as np
import IPython
import cPickle as pickle
import numpy.linalg as LA
import rospy


def list2str(deltas):
    label = " "
    for i in range(len(deltas)):
        label = label+str(deltas[i])+" "
    label = label+"\n"
    return label


class Common:   

    def __init__(self):
        print 'init'    


    #Common decorator that checks for override, else throws AssertionError
    def overrides(super_class):
        def overrider(method):
            assert(method.__name__ in dir(super_class))
            return method
        return overrider


    def rescale(self,deltas):
        ''''
        Rescale the deltas value, which are outputed from the neural network 
        to the original values

        Parameters
        ----------
        deltas: (3,) shape or (4,shape) numpy array 
            The deltas value should be either (3,) shape (x,y, theta) or 
            (x,y,theta,z)

        Returns
        -------
        (3,) or (4,) shape numpy array
            The subsequent scaled values. (x,y) are in pixel space, theta and z are in 
            robot pose
        '''
        if(len(deltas) < 3 or len(deltas) > 4):
            raise Exception("Delta is Not The Correct Size")
        deltas[0] = float(deltas[0])
        deltas[1] = float(deltas[1])
        deltas[2] = float(deltas[2])
        deltas[0] = self.constants[0]*deltas[0]+self.constants[2]
        deltas[1] = self.constants[1]*deltas[1]+self.constants[3]
        deltas[2] = (deltas[2]+1)*((self.Options.ROT_MAX - self.Options.ROT_MIN)/2)+self.Options.ROT_MIN
        if(len(deltas) == 4):
            deltas[3] = (deltas[3]+1)*((self.Options.Z_MAX - self.Options.Z_MIN)/2)+self.Options.Z_MIN
        return deltas


    def get_range(self):
        ''''
        Converts the bounding box specifed in robot pose, into the bounding
        box specifed in pixel space

        Returns
        -------
        list size 4
            First two elements specify the length of each range (x and y)
            Last two elementes specifiy the midpoint of each range
        '''
        low_bound = np.array([self.Options.X_LOWER_BOUND,self.Options.Y_LOWER_BOUND])
        up_bound = np.array([self.Options.X_UPPER_BOUND,self.Options.Y_UPPER_BOUND])

        #Convert to Pixels
        low_bound = self.reg.robot_to_pixel(low_bound)
        up_bound = self.reg.robot_to_pixel(up_bound)

        x_mid_range = (up_bound[0] - low_bound[0])/2.0
        y_mid_range = (up_bound[1] - low_bound[1])/2.0

        x_center = low_bound[0] + x_mid_range
        y_center = low_bound[1] + y_mid_range
        return [x_mid_range,y_mid_range,x_center,y_center]


    #Common but override
    def go_to_initial_state(self,yumi):
        #Takes arm out of camera field of view to record current state of the enviroment
        state = YuMiState([51.16, -99.4, 21.57, -107.19, 84.11, 94.61, -36.00])
        yumi.right.goto_state(state)
        #Open Gripper


    def eval_label(self,state):
        pos = self.rescale(state)
        #convert to robot frame 
        pos_p = self.reg.pixel_to_robot(pos[0:2])
        label = np.array([pos_p[0],pos_p[1],pos[2]])
        return label


    def eval_policy(self,state):
        outval = self.net.output(self.sess, state,channels=1)
        pos = self.rescale(outval)
        print "PREDICTED CORRECTION ", pos
        #print "PREDICTED POSE ", pos[2]
        return pos


    #Common
    def apply_deltas(self,delta_state,pose,grip_open,theta):
        """
            Get current states and apply given deltas
            Handle max and min states as well
        """
        g_open = grip_open
        if delta_state[3] != 0:
            g_open = delta_state[3]
        pose,theta = self.bound_pose(pose,theta, delta_state)
        theta = theta + delta_state[4]
        return pose, g_open,theta


    def next_evaluation(self):
        """
        :return: the String name of the next new potential rollout
                (i.e. do not overwrite another rollout)
        """
        i = 0
        prefix = self.Options.evaluations_dir
        path = prefix + 'rollout'+str(i) + "/"
        while os.path.exists(path):
            i += 1
            path = prefix + 'rollout'+str(i) + "/"
        return 'rollout' + str(i)


    def grip_open(self,gripper):
        try:
            gripper.command(1.2)
        except:
            rospy.logerr('grasp open error')


    def grip_squeeze(self,gripper):
        try:
            gripper.grasp(-0.1)
        except:
            rospy.logerr('grasp close error')


    def fix_buffer(self,data):
        frame_offset = 5
        num_state = len(data)
        cleaned_state = []
        for i in range(num_state-frame_offset):
            state_f = data[i+frame_offset]
            state_p = data[i]
            state_p['color_img'] = state_f['color_img']
            state_p['depth_img'] = state_f['depth_img']
            cur_action = state_p['action']
            if(LA.norm(cur_action) > -1e-3):
                print "GOT ACCEPTED"
                cleaned_state.append(state_p)
        return cleaned_state


    def save_recording(self,recording,rollouts=False):
        """Saves the recoring to the specified file.

        Parameters
        ----------
        recording: list 
            The recording of the label point shoud be a list of images and labels

        bc: BinaryCamera

        rollouts: bool 
            If True will save to a rollout directory instead of Supervisor (Default False)
        """
        recording = self.fix_buffer(recording)
        name = self.next_rollout()
        path = self.Options.rollouts_dir + name + '/'
        print "Saving to " + path + "..."
        os.makedirs(path)
        pickle.dump(recording,open(path+'rollout.p','wb'))
        print "Done saving."


    def save_evaluation(self,evaluation,rollouts=False):
        """Saves the recoring to the specified file.

        Parameters
        ----------
        recording: list 
            The recording of the label point shoud be a list of images and labels

        bc: BinaryCamera

        rollouts: bool 
            If True will save to a rollout directory instead of Supervisor (Default False)
        """
        #recording = self.fix_buffer(recording)
        name = self.next_evaluation()
        path = self.Options.evaluations_dir + name + '/'
        print "Saving to " + path + "..."
        os.makedirs(path)
        pickle.dump(evaluation,open(path+'rollout.p','wb'))
        print "Done saving."
