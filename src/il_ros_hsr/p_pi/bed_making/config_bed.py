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
DATA_PATH    = join(ROOT_DIR, 'collect_data_bed/')
ROLLOUT_PATH = join(DATA_PATH, 'rollouts/')

# Which blanket are we using?
BLANKET = 'white'
assert BLANKET in ['white', 'teal', 'cal']

# Put data here for _results_, i.e., from deployment.
RESULTS_PATH    = join(ROOT_DIR, 'results/')
DEPLOY_NET_PATH = join(RESULTS_PATH, 'deploy_network_{}'.format(BLANKET))
DEPLOY_ANA_PATH = join(RESULTS_PATH, 'deploy_analytic')
DEPLOY_HUM_PATH = join(RESULTS_PATH, 'deploy_human')

# STANDARD (the way I was doing earlier), CLOSE (the way they want).
# Update: change STANDARD to now use the CLOSE joints, but _original_ positions.
VIEW_MODE = 'standard'
assert VIEW_MODE in ['standard', 'close']

# ---------------------
# --- GRASP NETWORK ---
# ---------------------
# When deploying, we need to load in a config (text) file and a network.
g_data_name = 'cache_combo_v01'

# What I NORMALLY use, for depth images (though I also have a net trained for 8k steps).
g_head_name = 'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_6000_cv_False'
g_ckpt_name = '08_25_19_05_56_save.ckpt-6000'
g_conf_name = 'config_2018_08_25_19_00.txt'

# If I want RGB, use this trained network.
#g_head_name = 'grasp_1_img_rgb_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_8001_cv_False'
#g_ckpt_name = '09_05_08_31_04_save.ckpt-8000'
#g_conf_name = 'config_2018_09_05_08_24.txt'

## # And for Honda's newer data. Note the data name ...
## g_data_name = 'cache_combo_v03'
## g_head_name = 'grasp_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_8000_cv_False'
## g_ckpt_name = '09_11_18_18_50_save.ckpt-8000'
## g_conf_name = 'config_2018_09_11_18_13.txt'

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

# -----------------------
# --- SUCCESS NETWORK ---
# -----------------------
# What I normally use:
s_data_name = 'cache_combo_v01_success'
s_head_name = 'success_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_3000_cv_False'
s_ckpt_name = '08_28_11_44_47_save.ckpt-3000'
s_conf_name = 'config_2018_08_28_11_42.txt'

## # And for Honda's newer data. Note the data name ...
## s_data_name = 'cache_combo_v03_success'
## s_head_name = 'success_1_img_depth_opt_adam_lr_0.0001_L2_0.0001_kp_1.0_steps_5000_cv_False'
## s_ckpt_name = '09_11_19_51_00_save.ckpt-5000'
## s_conf_name = 'config_2018_09_11_19_47.txt'

SUCC_NET_PATH  = join(ROOT_DIR, 'success', s_data_name, s_head_name, s_ckpt_name)
SUCC_CONF_PATH = join(ROOT_DIR, 'success', s_data_name, s_head_name, s_conf_name)
assert 'save.ckpt' in s_ckpt_name and 'config' in s_conf_name

# Set SUCC_CONFIG to be the same as what we had during neural net training.
SUCC_CONFIG = BuildConfig()
with open(SUCC_CONF_PATH, 'r') as f:
    s_content = f.readlines()
s_content = [x.strip() for x in s_content]
for line in s_content:
    line = line.split(':')
    assert len(line) == 2
    attr = line[0].strip()
    value = line[1].strip()
    if value == 'True':
        setattr(SUCC_CONFIG, attr, True)
    if value == 'False':
        setattr(SUCC_CONFIG, attr, False)
    else:
        setattr(SUCC_CONFIG, attr, convert(value))


# --- Other stuff which I don't need to look at often ---

# Stuff for the tensioner. MAX_PULLS =/= GRASP_ATTEMPTS_PER_SIDE.
FORCE_LIMT = 15.0
HIGH_FORCE = 15.0
LOW_FORCE = 2.0
MAX_PULLS = 3

# For projecting a 2D pixel point into the 3D scene.
BOX = 10

# Max number of grasps to attempt _per_side_ before exiting.
GRASP_ATTEMPTS_PER_SIDE = 4

# Whether we use a sampler which tells us how to adjust the bed. (Not really used now)
INS_SAMPLE = False

# Gripper height. If these are off try tuning it. Michael used 0.065 (m).
# Update: hmm ... 0.055 still seems a bit off for our white sheet?
# Ack, 0.052 is resulting in a lot of 'hard' grasps where the robot grabs the
# sheet but also a bit of the blue surface underneath, so there's too much force.
# If 0.055 result in the robot 'missing' the sheet then we better adjust it
# dynamically? (UPDATE: forget these just dynamically do it ...)

GRIPPER_HEIGHT = 0.049
if BLANKET == 'cal':
    GRIPPER_HEIGHT += 0.002 # I think the sheet is thinner, we need to be safe.
elif BLANKET == 'teal':
    GRIPPER_HEIGHT -= 0.002 # I think Teal is thicker, we can afford to grasp lower.


# TODO: we should probably have this so we know if depth is in mm or meters?
# Right now for data preprocessing I just use 'HSR' because the code has so many
# HSR-dependent stuff, but might be worth using in the future.
ROBOT = 'HSR'
assert ROBOT in ['HSR', 'Fetch']

# TODO: it's millimeters to meters but should understand code usage.
# Michael needed this for computing grasp poses.
MM_TO_M = 0.001

# For the gripper code. Always keep at False for now.
USE_DART = False

# ------------------------------------------------------------------------------
# OLDER STUFF I'LL RESOLVE LATER




# CLASSES = ['success_bed','failure_bed']
# 
# # path and dataset parameter
# 

# if USE_DART: 
# 	ROLLOUT_PATH = DATA_PATH+'rollouts_dart_cal/'
# 	BC_HELD_OUT = DATA_PATH+'held_out_debugpython'
# else: 
# 	BC_HELD_OUT = DATA_PATH+'held_out_bc'
# FAST_PATH = DATA_PATH+'fast_pic/'
# 
 
# GRASP_LABEL_PATH = DATA_PATH+'grasp_labels/'
# SUCCESS_LABEL_PATH = DATA_PATH+'success_labels/'
# 
# TRAN_OUTPUT_DIR = DATA_PATH +'transition_output/' 
# TRAN_STATS_DIR = TRAN_OUTPUT_DIR + 'stats/'
# TRAIN_STATS_DIR_T = TRAN_OUTPUT_DIR + 'train_stats/'
# TEST_STATS_DIR_T = TRAN_OUTPUT_DIR + 'test_stats/'
# 
# GRASP_OUTPUT_DIR = DATA_PATH + 'grasp_output/'
# GRASP_STAT_DIR = GRASP_OUTPUT_DIR + 'rollout_cs/' 
# TRAIN_STATS_DIR_G = GRASP_OUTPUT_DIR + 'train_stats/'
# TEST_STATS_DIR_G = GRASP_OUTPUT_DIR + 'test_stats/'
# 
# PRE_TRAINED_DIR = '/home/autolab/Workspaces/michael_working/yolo_tensorflow/data/pascal_voc/weights/'
# WEIGHTS_FILE = None
# # WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')
# 
# # (Daniel note: michael had to switch these around when we were testing)
# if USE_DART:
# 	# GRASP_NET_NAME = '11_08_15_47_03_CS_0_save.ckpt-500'
# 	# TRAN_NET_NAME = '11_08_16_01_28_CS_0_save.ckpt-1000'
# 	GRASP_NET_NAME = '02_10_17_29_27_CS_0_save.ckpt-1000'
# 	TRAN_NET_NAME = '02_07_16_52_00_CS_0_save.ckpt-1000'
# else:
# 	#BC_NETWORK
# 	TRAN_NET_NAME = "09_06_00_10_12_SS_0save.ckpt-30300"
# 
# 	#BC_NETWORK 
# 	GRASP_NET_NAME = "09_09_12_01_49_CS_0_save.ckpt-1200"
# 
# #GRASP_NET_NAME = "09_08_11_14_12_CS_1_save.ckpt-6000"
# 
# MSR_LOSS = True
# 
# RIGHT_SIDE = True
# 
# #DEBUG
# DEBUG_MODE = False
# 
