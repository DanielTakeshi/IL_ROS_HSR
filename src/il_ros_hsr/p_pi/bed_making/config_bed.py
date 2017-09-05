import os

#
# path and dataset parameter
#

ROOT_DIR = '/media/autolab/1tb/data/'

NET_NAME = '07_31_00_09_46save.ckpt-30300'
DATA_PATH = ROOT_DIR + 'bed_rcnn/'

ROLLOUT_PATH = DATA_PATH+'rollouts/'
STAT_PATH = DATA_PATH+'stats/'
GRASP_LABEL_PATH = DATA_PATH+'grasp_labels/'
SUCCESS_LABEL_PATH = DATA_PATH+'success_labels/'

PRE_TRAINED_DIR = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/'

WEIGHTS_FILE = None


# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['yes','no']

# #CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
# 			'train', 'tvmonitor']




USE_WEB_INTERFACE = False

#SS LEARN
SS_LEARN = True
NUM_SS_DATA = 3
SS_TIME = 1.2

#TENSIONER 
FORCE_LIMT = 30.0
HIGH_FORCE = 15.0
LOW_FORCE = 2.0
MAX_PULLS = 6

#DEBUG
DEBUG_MODE = False


#GRIPPER 
GRIPPER_HEIGHT = 0.060
MM_TO_M = 0.001

INS_SAMPLE = True