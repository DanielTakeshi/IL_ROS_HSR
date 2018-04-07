import os
import numpy as np

#
# path and dataset parameter
#

ROOT_DIR = '/media/autolab/1tb/data/'

NET_NAME = '07_31_00_09_46save.ckpt-30300'
DATA_PATH = ROOT_DIR + 'bed_rcnn/'


USE_DART = False


if USE_DART: 

	ROLLOUT_PATH = DATA_PATH+'rollouts_dart_cal/'

	BC_HELD_OUT = DATA_PATH+'held_out_cal'
else: 
	ROLLOUT_PATH = DATA_PATH+'rollouts/'

	BC_HELD_OUT = DATA_PATH+'held_out_bc'

FAST_PATH = DATA_PATH+'fast_pic/'


#STAT_PATH = DATA_PATH+'stats_dart_adv/'
STAT_PATH = DATA_PATH+'stats/'
#STAT_PATH = DATA_PATH+'stats/'
#STAT_PATH = DATA_PATH+'stats_analytic_adv/'



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



if USE_DART:
	GRASP_NET_NAME = "10_17_16_25_45_CS_0_save.ckpt-500"
	TRAN_NET_NAME = "10_17_16_33_07_CS_0_save.ckpt-500"
else:
	#BC_NETWORK
	TRAN_NET_NAME = "09_06_00_10_12_SS_0save.ckpt-30300"

	#BC_NETWORK 
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
FORCE_LIMT = 25.0
HIGH_FORCE = 25.0
LOW_FORCE = 2.0
MAX_PULLS = 3
BOX = 10

#DEBUG
DEBUG_MODE = False


#GRIPPER 
GRIPPER_HEIGHT = 0.06
#GRIPPER_HEIGHT = 0.090
MM_TO_M = 0.001



GRASP_OUT = 8
INS_SAMPLE = True


DART_MAT = np.array([[ 1421.21439203,  -158.39422591],
 					[ -158.39422591, 165.80726958]])