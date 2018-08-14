import os, sys
import numpy as np
from os.path import join


# ------------------------------------------------------------------------------

# Overall root directory where we save files. Should be /nfs/diskstation.
ROOT_DIR = '/nfs/diskstation/seita/bed-make/'

# If we are running the `collect_data_bed` scripts, we save rollouts here.
DATA_PATH = join(ROOT_DIR, 'collect_data_bed/')
ROLLOUT_PATH = join(DATA_PATH, 'rollouts/')
# There is also a BC_HELD_OUT but we probably don't need this.

# STANDARD (the way I was doing earlier), CLOSE (the way they want).
VIEW_MODE = 'close'
assert VIEW_MODE in ['standard', 'close']

# Whether we use a sampler which tells us how to adjust the bed. (Not really used now)
INS_SAMPLE = False

# Stuff for the tensioner.
FORCE_LIMT = 25.0
HIGH_FORCE = 25.0
LOW_FORCE = 2.0
MAX_PULLS = 3
BOX = 10

# Max number of grasps to attempt before exiting.
GRASP_OUT = 8

# ------------------------------------------------------------------------------
# OLDER STUFF I'LL RESOLVE LATER


CLASSES = ['success_bed','failure_bed']

# path and dataset parameter
NET_NAME = '07_31_00_09_46save.ckpt-30300'

USE_DART = False
if USE_DART: 
	ROLLOUT_PATH = DATA_PATH+'rollouts_dart_cal/'
	BC_HELD_OUT = DATA_PATH+'held_out_debugpython'
else: 
	BC_HELD_OUT = DATA_PATH+'held_out_bc'
FAST_PATH = DATA_PATH+'fast_pic/'

#STAT_PATH = DATA_PATH+'stats_dart_adv/'
STAT_PATH = DATA_PATH+'stats_cal_debug/'
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

# (Daniel note: michael had to switch these around when we were testing)
if USE_DART:
	# GRASP_NET_NAME = '11_08_15_47_03_CS_0_save.ckpt-500'
	# TRAN_NET_NAME = '11_08_16_01_28_CS_0_save.ckpt-1000'
	GRASP_NET_NAME = '02_10_17_29_27_CS_0_save.ckpt-1000'
	TRAN_NET_NAME = '02_07_16_52_00_CS_0_save.ckpt-1000'
else:
	#BC_NETWORK
	TRAN_NET_NAME = "09_06_00_10_12_SS_0save.ckpt-30300"

	#BC_NETWORK 
	GRASP_NET_NAME = "09_09_12_01_49_CS_0_save.ckpt-1200"

#GRASP_NET_NAME = "09_08_11_14_12_CS_1_save.ckpt-6000"

MSR_LOSS = True

RIGHT_SIDE = True

#DEBUG
DEBUG_MODE = False

#GRIPPER 
GRIPPER_HEIGHT = 0.055
#GRIPPER_HEIGHT = 0.065 # daniel: original value
#GRIPPER_HEIGHT = 0.090
MM_TO_M = 0.001

