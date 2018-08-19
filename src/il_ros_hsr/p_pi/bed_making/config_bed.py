import os, sys
import numpy as np
from os.path import join

# Placeholder class that we use for feeding in attributes to simulate a config.
class BuildConfig:
    pass

def convert(s):
    """ Need this b/c configs are stored as strings but some values are floats/ints.

    However, be careful, if `s` is a True/False value, it will return 1.0/0.0, resp.
    But, since the booleans will evaluate in a similar way, I _think_ this is OK...
    Also, be careful about ints vs floats. We need some values as integers...
    Update: for now let's handle the True/False cases beforehand.
    """
    try:
        return float(s)
    except:
        return s

# ------------------------------------------------------------------------------

# Overall root directory where we save files. Should be /nfs/diskstation.
ROOT_DIR = '/nfs/diskstation/seita/bed-make/'

# If we are running the `collect_data_bed` scripts, we save rollouts here.
DATA_PATH = join(ROOT_DIR, 'collect_data_bed/')
ROLLOUT_PATH = join(DATA_PATH, 'rollouts/')

# STANDARD (the way I was doing earlier), CLOSE (the way they want).
# Update: change STANDARD to now use the CLOSE joints, but _original_ positions.
VIEW_MODE = 'standard'
assert VIEW_MODE in ['standard', 'close']

# When deploying, we need to load in a config (text) file and a network.
g_data_name = 'cache_white_v01'
g_head_name = 'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_cv_False'
g_ckpt_name = '08_14_18_24_24_save.ckpt-2500' 
g_conf_name = 'config_2018_08_14_18_19.txt'
GRASP_NET_PATH  = join(ROOT_DIR, 'grasp', g_data_name, g_head_name, g_ckpt_name)
GRASP_CONF_PATH = join(ROOT_DIR, 'grasp', g_data_name, g_head_name, g_conf_name)
assert 'save.ckpt' in g_ckpt_name and 'config' in g_conf_name

# Set GRASP_CONFIG to be the same as what we had during neural net training.
GRASP_CONFIG = BuildConfig()
with open(GRASP_CONF_PATH, 'r') as f:
    g_content = f.readlines()
g_content = [x.strip() for x in g_content]
for line in g_content:
    line = line.split(':')
    assert len(line) == 2
    attr = line[0].strip()
    value = line[1].strip()
    if value == 'True':
        setattr(GRASP_CONFIG, attr, True)
    if value == 'False':
        setattr(GRASP_CONFIG, attr, False)
    else:
        setattr(GRASP_CONFIG, attr, convert(value))

# Do the same for the success network ...
# ...
# ...


# --- Other stuff which I don't need to look at often ---

# Stuff for the tensioner.
FORCE_LIMT = 25.0
HIGH_FORCE = 25.0
LOW_FORCE = 2.0
MAX_PULLS = 3
BOX = 10

# Max number of grasps to attempt before exiting.
GRASP_OUT = 8

# Whether we use a sampler which tells us how to adjust the bed. (Not really used now)
INS_SAMPLE = False

# Gripper height. If these are off try tuning it. Michael used 0.065 (m).
GRIPPER_HEIGHT = 0.055


# ------------------------------------------------------------------------------
# OLDER STUFF I'LL RESOLVE LATER




CLASSES = ['success_bed','failure_bed']

# path and dataset parameter

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

MM_TO_M = 0.001

