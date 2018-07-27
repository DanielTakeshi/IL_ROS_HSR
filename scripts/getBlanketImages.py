import threading

import control_msgs.msg
import cv2 as cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Joy
import trajectory_msgs.msg
import math
import datetime
import pickle

pubArm = rospy.Publisher('/hsrb/arm_trajectory_controller/command',
                         trajectory_msgs.msg.JointTrajectory, queue_size=1)
pubHead = rospy.Publisher('/hsrb/head_trajectory_controller/command',
                          trajectory_msgs.msg.JointTrajectory, queue_size=1)


class ImageC(object):
    def __init__(self):
        # event that will block until the info is received
        self._event = threading.Event()
        # attribute for storing the rx'd message
        self._msg = None

    def __call__(self, headImage):
        # Uses __call__ so the object itself acts as the callback
        # save the data, trigger the event
        self._msg = headImage
        self._event.set()

    def get_im(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        # self._event.wait(timeout)
        return self._msg


class HeadPos(object):
    def __init__(self):
        # event that will block until the info is received
        self._event = threading.Event()
        # attribute for storing the rx'd message
        self._msg = None

    def __call__(self, pos):
        # Uses __call__ so the object itself acts as the callback
        # save the data, trigger the event
        p = pos.actual.positions
        self._msg = p
        self._event.set()

    def get_armFlexPos(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        # self._event.wait(timeout)
        return self._msg


class JoyButtons(object):
    def __init__(self):
        # event that will block until the info is received
        self._event = threading.Event()
        # attribute for storing the rx'd message
        self._msg = None

    def __call__(self, btn):
        # Uses __call__ so the object itself acts as the callback
        # save the data, trigger the event
        self._msg = btn.buttons
        self._event.set()

    def get_btn(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        # self._event.wait(timeout)
        return self._msg


class TrajStat(object):
    def __init__(self):
        # event that will block until the info is received
        self._event = threading.Event()
        # attribute for storing the rx'd message
        self._msg = None

    def __call__(self, state):
        # Uses __call__ so the object itself acts as the callback
        # save the data, trigger the event
        self._msg = state.actual.positions
        self._event.set()

    def get_state(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        # self._event.wait(timeout)
        return self._msg


def OnShutdown_callback():
    global isRunning
    isRunning = False


RGBImageObj = ImageC()
depthImageObj = ImageC()
headStateObj = HeadPos()
joyBtnObj = JoyButtons()
armState = TrajStat()
headState = TrajStat()

windowName = 'main'


def moveArm(ALJ1, AFJ1, ARJ1, WFJ1, WRJ1):
    # moving the arm
    try:
        while pubArm.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.01)
    except KeyboardInterrupt:
        rospy.loginfo(KeyboardInterrupt)
    if not isRunning:
        raise ValueError
    traj = trajectory_msgs.msg.JointTrajectory()
    traj.joint_names = ["arm_lift_joint", "arm_flex_joint",
                        "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
    p = trajectory_msgs.msg.JointTrajectoryPoint()
    p.positions = [ALJ1, AFJ1, ARJ1, WFJ1, WRJ1]
    # p.positions = [0, 0, 3.14 / 2, -3.14 / 2, 0]
    p.velocities = [0, 0, 0, 0, 0]
    p.time_from_start = rospy.Time(0.5)
    traj.points = [p]
    pubArm.publish(traj)


def moveHead(HTJ, HPJ):
    try:
        while pubArm.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.01)
    except KeyboardInterrupt:
        rospy.loginfo(KeyboardInterrupt)
    if not isRunning:
        raise ValueError
    traj = trajectory_msgs.msg.JointTrajectory()
    traj.joint_names = ["head_tilt_joint", "head_pan_joint"]
    p = trajectory_msgs.msg.JointTrajectoryPoint()
    p.positions = [HTJ, HPJ]
    # p.positions = [0, 0, 3.14 / 2, -3.14 / 2, 0]
    p.velocities = [0, 0]
    p.time_from_start = rospy.Time(0.5)
    traj.points = [p]
    pubHead.publish(traj)


def creatingConCatImages(RGBImage, depthImage):
    edges = cv2.Canny(RGBImage, 100, 200)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    minI, maxI, _, _ = cv2.minMaxLoc(
        np.reshape(depthImage, -1))  # get min,max values
    normDepthImage = (depthImage - minI) / (maxI - minI) * \
        255  # create normalized image
    # normDepthImage = np.floor(normDepthImage)
    normDepthImage = normDepthImage.astype(np.uint8)

    normDepthImage_a = cv2.bitwise_and(
        normDepthImage, cv2.bitwise_not(edges), mask=None)
    normDepthImage_b = cv2.add(normDepthImage_a, edges)
    comb2 = cv2.merge((normDepthImage_a, normDepthImage_a, normDepthImage_b))

    height, width = depthImage.shape
    # cv2.imwrite('tmp.jpg', comb2)
    div = np.zeros((height, 30, 3), np.uint8)
    div[:, :, 0] = 255
    conImage = np.concatenate((RGBImage, comb2), axis=1)
    # conImage = np.concatenate((conImage,np.zeros((height,width*2,3),np.uint8)),axis=0)
    # conImage=np.concatenate((conImage,comb2),axis=1)
    cv2.imshow(windowName, conImage)

    # cv2.imshow('depth', normDepthImage)
    # cv2.imshow('edges', edges)
    # cv2.imshow('conImage', conImage)
    cv2.waitKey(20)
    return conImage
    # cv2.imshow('depth', depthImage)
    # cv2.waitKey(3)
    # print(depthImage.shape[:])
    # print(depthImage.dtype)
    # print(depthImage[100,100])
    # btn = joyBtnObj.get_btn()
    # n = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # data = {"ULx": ULx,
    #         "ULy": ULy,
    #         "LRx": LRx,
    #         "LRy": LRy,
    #         "platformPosX": platformPosX,
    #         "platformPosY": platformPosY,
    #         "platformPosZ": platformPosZ,
    #         "endPlatformPosX": endPlatformPosX,
    #         "endPlatformPosY": endPlatformPosY,
    #         "endPlatformPosZ": endPlatformPosZ,
    #         "platformTrajX": platformTrajX,
    #         "platformTrajY": platformTrajY,
    #         "platformTrajZ": platformTrajZ,
    #         "initArmPos": initArmPos,
    #         "armEndPos": armEndPos,
    #         "inputIm": im}
    # try:
    #     os.makedirs("dataExp")
    # except:
    #     pass
    # pickle.dump(data, open("dataExp/"+n+".p", "wb"))

    # cv2.imshow('tmp',1)
    # cv2.waitKey(-1)
    # rospy.sleep(1)


def main():
    global isRunning
    isRunning = True

    rospy.init_node('main', anonymous=True)
    rospy.on_shutdown(OnShutdown_callback)
    rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw",
                     Image, RGBImageObj, queue_size=1)
    rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image",
                     Image, depthImageObj, queue_size=1)
    rospy.Subscriber("/hsrb/head_trajectory_controller/state",
                     control_msgs.msg.JointTrajectoryControllerState, headStateObj, queue_size=1)
    rospy.Subscriber("/hsrb/joy", Joy, joyBtnObj, queue_size=1)
    rospy.Subscriber("/hsrb/arm_trajectory_controller/state",
                     control_msgs.msg.JointTrajectoryControllerState, armState, queue_size=1)
    rospy.Subscriber("/hsrb/head_trajectory_controller/state",
                     control_msgs.msg.JointTrajectoryControllerState, headState, queue_size=1)

    rospy.sleep(2)
    for H in range(25,51,5):
        for A in [45,61]:
            print("Height={} Head_angle={}".format(H,A))
            moveHead(np.deg2rad(-A), 0)
            moveArm(float(H)/100, -2.5, 0, -math.pi / 2, -math.pi / 2)

            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            bridge = CvBridge()
            RGBImage = bridge.imgmsg_to_cv2(RGBImageObj.get_im(), "bgr8")
            depthImage = bridge.imgmsg_to_cv2(depthImageObj.get_im(), "32FC1")
            conImage = creatingConCatImages(RGBImage, depthImage)
            height, width, _ = conImage.shape

            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height * 2))
            # video = cv2.VideoWriter('video.avi',-1,1,(width,height))
            n=0
            while isRunning and n<=19:
                RGBImage = bridge.imgmsg_to_cv2(RGBImageObj.get_im(), "bgr8")
                depthImage = bridge.imgmsg_to_cv2(depthImageObj.get_im(), "32FC1")
                # conImage = creatingConCatImages(RGBImage, depthImage)
                # video.write(conImage)
                btn = joyBtnObj.get_btn()
                # print(,)
                if btn[14]:
                    dirPath = "/home/ron/Desktop/cornerDetectionImages/"
                    now = datetime.datetime.now()
                    timeStr = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"_"+str(now.hour)+":"+str(now.minute)+":"+str(now.second)+":"+str(now.microsecond)
                    outData = {
                        "RGBImage":RGBImage,
                        "depthImage":depthImage,
                        "armState":armState.get_state(),
                        "headState":headState.get_state(),
                        "markerPos":None
                    }
                    pickle.dump(outData,open(dirPath+timeStr+".pkl","wb"))
                    rospy.sleep(0.1)
                    n=n+1
                    cv2.imshow("main",RGBImage)
                    cv2.waitKey(3)
            cv2.destroyAllWindows()
            # video.release()


if __name__ == '__main__':
    main()
