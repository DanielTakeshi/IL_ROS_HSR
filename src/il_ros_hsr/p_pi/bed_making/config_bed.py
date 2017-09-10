import os

#
# path and dataset parameter
#

ROOT_DIR = '/media/autolab/1tb/data/'

NET_NAME = '07_31_00_09_46save.ckpt-30300'
DATA_PATH = ROOT_DIR + 'bed_rcnn/'

ROLLOUT_PATH = DATA_PATH+'rollouts/'

BC_HELD_OUT = DATA_PATH+'held_out_bc'
#STAT_PATH = DATA_PATH+'stats_adverserial/'
STAT_PATH = DATA_PATH+'stats/'
GRASP_LABEL_PATH = DATA_PATH+'grasp_labels/'
SUCCESS_LABEL_PATH = DATA_PATH+'success_labels/'

TRAN_OUTPUT_DIR = DATA_PATH +'transition_output/' 
TRAN_STATS_DIR = TRAN_OUTPUT_DIR + 'stats/'
TRAIN_STATS_DIR_T = TRAN_OUTPUT_DIR + 'train_stats/'
TEST_STATS_DIR_T = TRAN_OUTPUT_DIR + 'test_stats/'


GRASP_OUTPUT_DIR = DATA_PATH + 'grasp_output/'
GRASP_STAT_DIR = GRASP_OUTPUT_DIR + 'rollout_cs/' 
TRAIN_STATS_DIR_G = GRASP_OUTPUT_DIR + 'train_stats/'
TEST_STATS_DIR_G = GRASP_OUTPUT_DIR + 'test_stats/'

PRE_TRAINED_DIR = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/'

WEIGHTS_FILE = None


# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['yes','no']

# #CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
# 			'train', 'tvmonitor']


TRAN_NET_NAME = "09_06_00_10_12_SS_0save.ckpt-30300"
GRASP_NET_NAME = "09_09_12_01_49_CS_0_save.ckpt-1200"
#GRASP_NET_NAME = "09_08_11_14_12_CS_1_save.ckpt-6000"


MSR_LOSS = True

USE_WEB_INTERFACE = False

#SS LEARN
SS_LEARN = False
NUM_SS_DATA = 3
SS_TIME = 1.2


RIGHT_SIDE = True


#TENSIONER 
FORCE_LIMT = 30.0
HIGH_FORCE = 15.0
LOW_FORCE = 2.0
MAX_PULLS = 6
BOX = 5

#DEBUG
DEBUG_MODE = False


#GRIPPER 
GRIPPER_HEIGHT = 0.060
MM_TO_M = 0.001

INS_SAMPLE = True